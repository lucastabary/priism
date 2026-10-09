"""Keep the separator good on real music: distillation from the pretrained BS-RoFormer on real songs.

Fine-tuning on synthetic songs alone makes the core forget part of what it knew about real recordings:
on NI stems, BS-Roformer-SW as is beats our fine-tuned separator once both are grouped into the same
stems. So some training steps use real songs without stems (FMA): the frozen pretrained model (the
teacher) splits each crop into its fixed stems (bass, drums, other, vocals, guitar, piano), each of
our outputs is assigned to the teacher stem it explains best, and the grouped outputs are trained to
match the teacher's stems. Our finer split (one track per instrument) stays free inside each group.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .loss import neg_snr

AUDIO = (".mp3", ".flac", ".wav", ".ogg")


class RealCrops(IterableDataset):
    """Endless random ``chunk_s`` crops of real songs (any audio files under ``folder``), stereo, mix only."""

    def __init__(self, folder: str | Path, chunk_s: float, sample_rate: int, seed: int = 0, min_rms: float = 1e-3):
        self.files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO)
        if not self.files:
            raise ValueError(f"no audio files under {folder}")
        self.chunk = int(chunk_s * sample_rate)
        self.sr = sample_rate
        self.seed = seed
        self.min_rms = min_rms

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng((self.seed, info.id if info else 0, os.getpid(), time.time_ns()))
        while True:
            path = self.files[int(rng.integers(len(self.files)))]
            try:
                with sf.SoundFile(path) as f:
                    if f.samplerate != self.sr or f.frames < self.chunk:
                        continue
                    f.seek(int(rng.integers(0, f.frames - self.chunk)))
                    x = f.read(self.chunk, dtype="float32", always_2d=True)
            except (RuntimeError, OSError):  # an unreadable file is skipped, not fatal
                continue
            if x.shape[0] < self.chunk:  # mp3 lengths are estimates: a crop near the end can come out short
                continue
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            if np.sqrt(np.mean(x**2)) < self.min_rms:
                continue
            yield torch.from_numpy(np.ascontiguousarray(x[:, :2].T))


def stem_songs(folder: str | Path) -> list[list[Path]]:
    """Real multitracks: every folder under ``folder`` holding two or more stem files (mixture.* left out),
    e.g. MUSDB18-HQ (vocals, drums, bass, other) or MoisesDB-style folders (one file per stem)."""
    songs = {}
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() in AUDIO and f.stem.lower() not in ("mixture", "mix"):
            songs.setdefault(f.parent, []).append(f)
    return [v for v in songs.values() if len(v) >= 2]


class StemCrops(IterableDataset):
    """Endless random crops of real multitracks: (mix (C, S), stems (J, C, S)), mix = sum of the stems.
    Trained with ``group_loss`` against the true stems: our finer tracks stay free inside each stem."""

    def __init__(self, folder: str | Path, chunk_s: float, sample_rate: int, seed: int = 0, max_stems: int = 8,
                 min_rms: float = 1e-3):
        self.songs = stem_songs(folder)
        if not self.songs:
            raise ValueError(f"no multitrack folders under {folder}")
        self.chunk, self.sr, self.seed, self.max_stems, self.min_rms = int(chunk_s * sample_rate), sample_rate, seed, max_stems, min_rms

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng((self.seed, info.id if info else 0, os.getpid(), time.time_ns()))
        while True:
            files = self.songs[int(rng.integers(len(self.songs)))]
            try:
                infos = [sf.info(f) for f in files]
                frames = min(i.frames for i in infos)
                if any(i.samplerate != self.sr for i in infos) or frames < self.chunk:
                    continue
                a = int(rng.integers(0, frames - self.chunk))
                stems = []
                for f in files:
                    x, _ = sf.read(f, start=a, frames=self.chunk, dtype="float32", always_2d=True)
                    if x.shape[0] < self.chunk:
                        break
                    stems.append(np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x[:, :2])
                else:
                    y = np.stack(stems).transpose(0, 2, 1)  # (J, C, S)
                    if np.sqrt(np.mean(y.sum(0) ** 2)) < self.min_rms:
                        continue
                    if len(y) > self.max_stems:  # many-stem songs: the quietest stems join one group
                        order = np.argsort(-np.sqrt((y**2).mean((1, 2))))
                        y = np.concatenate([y[order[: self.max_stems - 1]], y[order[self.max_stems - 1:]].sum(0, keepdims=True)])
                    pad = np.zeros((self.max_stems - len(y),) + y.shape[1:], np.float32)  # silent stems drop out of the loss
                    y = np.concatenate([y, pad])
                    yield torch.from_numpy(y.sum(0)), torch.from_numpy(np.ascontiguousarray(y))
            except (RuntimeError, OSError):
                continue


def group_loss(est: torch.Tensor, refs: torch.Tensor, min_share: float = 1e-3) -> tuple[torch.Tensor, float]:
    """est (B, K, C, S) our outputs, refs (B, J, C, S) teacher stems -> (-SNR of grouped outputs, mean SNR).

    Each output joins the stem it overlaps most (largest inner product: the choice that lowers the grouped
    error most; decided without gradient). Stems that hold less than ``min_share`` of the crop's energy
    (the teacher's silent stems) take no output and are left out of the loss. Dividing by each stem's
    energy instead sent most outputs to near-silent stems, which held distillation at ~0 dB.
    """
    B, K = est.shape[:2]
    e = est.flatten(2)
    r = refs.flatten(2)
    with torch.no_grad():
        energy = r.pow(2).sum(-1)
        keep = energy > min_share * energy.sum(-1, keepdim=True)  # (B, J)
        overlap = torch.einsum("bks,bjs->bkj", e.float(), r.float()).masked_fill(~keep[:, None], float("-inf"))
        best = overlap.argmax(-1)  # (B, K)
        onehot = torch.nn.functional.one_hot(best, refs.shape[1]).to(est.dtype)  # (B, K, J)
    grouped = torch.einsum("bkj,bkcs->bjcs", onehot, est)
    snr = neg_snr(grouped, refs)[keep]
    loss = snr.mean() if snr.numel() else est.sum() * 0
    return loss, float(-loss.detach())


@torch.no_grad()
def teacher_stems(teacher: torch.nn.Module, mix: torch.Tensor, chunk: int = 2) -> torch.Tensor:
    """Teacher stems (B, J, C, S) in fp32, whatever autocast is active: the stock MSST model cannot run
    its complex mask in bf16. Two crops at a time keep fp32 attention within memory."""
    with torch.autocast(mix.device.type, enabled=False):
        return torch.cat([teacher(m.float()).float() for m in mix.split(chunk)])


def load_teacher(config: str | Path, ckpt: str | Path, msst_path: str | Path, device: str) -> torch.nn.Module:
    from .msst_core import load_msst_roformer

    teacher = load_msst_roformer(config, ckpt, msst_path).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher
