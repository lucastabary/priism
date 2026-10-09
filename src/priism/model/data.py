"""Training examples from songs written by ``priism gen``: random crops, one target per present source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SILENT_DB = -50.0  # a source this far below the crop's mix level counts as absent in the crop


def merged_sources(meta: dict) -> list[list[dict]]:
    """Groups of sources that form one target: a merge group (ambiguous pair) is a single target."""
    groups: dict[int, list[dict]] = {}
    for s in meta["sources"]:
        key = s["merge_group"] if s.get("merge_group") is not None else s["id"]
        groups.setdefault(key, []).append(s)
    return list(groups.values())


class SongChunks(Dataset):
    """Random ``chunk_s`` crops of generated songs. Item: (mix (2, S), targets (N, 2, S)), N = sources heard."""

    def __init__(self, folders: list[str | Path] | str | Path, chunk_s: float, sample_rate: int,
                 items_per_song: int = 8, seed: int = 0, lossy_p: float = 0.0, labels: bool = False):
        if isinstance(folders, (str, Path)):
            root = Path(folders)
            folders = sorted(p.parent for p in root.glob("*/meta.json")) or [root]
        self.songs = [Path(f) for f in folders]
        self.metas = [json.loads((f / "meta.json").read_text()) for f in self.songs]
        for m in self.metas:
            if m["sample_rate"] != sample_rate:
                raise ValueError(f"song at {m['sample_rate']} Hz, model at {sample_rate} Hz")
        self.chunk = int(chunk_s * sample_rate)
        self.sr = sample_rate
        self.items_per_song = items_per_song
        self.seed = seed
        self.lossy_p = lossy_p
        self.labels = labels  # items also carry what each source plays (see crop_example)

    def __len__(self) -> int:
        return len(self.songs) * self.items_per_song

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = i % len(self.songs)
        rng = np.random.default_rng((self.seed, i))
        return crop_example(self.songs[j], self.metas[j], self.chunk, self.sr, rng, self.lossy_p,
                            with_labels=self.labels)

    def item_with_families(self, i: int):
        """Item ``i`` plus the twin family of each target (see crop_example)."""
        j = i % len(self.songs)
        rng = np.random.default_rng((self.seed, i))
        return crop_example(self.songs[j], self.metas[j], self.chunk, self.sr, rng, self.lossy_p, with_families=True)


def crop_example(song: Path, meta: dict, chunk: int, sr: int, rng: np.random.Generator, lossy_p: float = 0.0,
                 with_families: bool = False, with_labels: bool = False):
    """A random ``chunk``-sample crop of one song: (mix (2, S), targets (N, 2, S)), N = sources heard.

    ``with_families`` adds, per heard target, the id of its twin family (the original's id, shared by all
    parts the same instrument plays; a lone source is its own family).
    ``with_labels`` adds a third element for the factor heads (model/factors.py): per heard target its
    notes in the crop (``timeline``, None for songs written before timelines), twin family and effects.
    """
    total = int(round(meta["duration_s"] * sr))
    start = int(rng.integers(0, max(1, total - chunk)))
    stop = start + chunk

    def read(path: Path) -> np.ndarray:
        x, _ = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
        return np.pad(x, ((0, chunk - len(x)), (0, 0))).T  # (2, S)

    targets = [sum(read(song / s["file"]) for s in group) for group in merged_sources(meta)]
    mix = np.sum(targets, axis=0)  # exactly the sum, as in the song
    level = np.sqrt(np.mean(mix**2)) + 1e-9
    loud = [20 * np.log10(np.sqrt(np.mean(t**2)) / level + 1e-12) > SILENT_DB for t in targets]
    heard = [t for t, ok in zip(targets, loud) if ok]
    fams = [g[0]["twin_of"] if g[0].get("twin_of") is not None and g[0].get("merge_group") is None else g[0]["id"]
            for g, ok in zip(merged_sources(meta), loud) if ok]
    if lossy_p:
        from ..gen.augment import random_lossy

        mix = random_lossy(mix.T, sr, rng, lossy_p)[0].T
    tg = np.stack(heard) if heard else np.zeros((0, *mix.shape), np.float32)
    out = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32)), torch.from_numpy(tg.astype(np.float32))
    if with_labels:
        from .factors import fx_vector

        groups = [g for g, ok in zip(merged_sources(meta), loud) if ok]
        t0, t1 = start / sr, stop / sr
        def notes(g):
            if any("timeline" not in s for s in g):
                return None
            tl = np.asarray([e for s in g for e in s["timeline"] if e[1] > t0 and e[0] < t1], np.float32).reshape(-1, 5)
            tl[:, :2] -= t0
            return tl
        timelines = [notes(g) for g in groups]
        lab = {"timeline": timelines if any(t is not None for t in timelines) else None, "families": fams,
               "fx": np.stack([fx_vector(g[0].get("fx")) for g in groups]) if groups else np.zeros((0, 11), np.float32)}
        return (*out, lab)
    return (*out, fams) if with_families else out


def collate(batch: list[tuple]) -> tuple:
    """(mixes, targets, n) and, when items carry labels, the list of labels as a fourth element."""
    mixes = torch.stack([x[0] for x in batch])
    n = torch.tensor([x[1].shape[0] for x in batch])
    nmax = max(int(n.max()), 1)
    tg = torch.zeros(len(batch), nmax, *mixes.shape[1:])
    for b, x in enumerate(batch):
        tg[b, : x[1].shape[0]] = x[1]
    if len(batch[0]) > 2:
        return mixes, tg, n, [x[2] for x in batch]
    return mixes, tg, n
