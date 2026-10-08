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
                 items_per_song: int = 8, seed: int = 0, lossy_p: float = 0.0):
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

    def __len__(self) -> int:
        return len(self.songs) * self.items_per_song

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        song, meta = self.songs[i % len(self.songs)], self.metas[i % len(self.songs)]
        rng = np.random.default_rng((self.seed, i))
        total = int(round(meta["duration_s"] * self.sr))
        start = int(rng.integers(0, max(1, total - self.chunk)))
        stop = start + self.chunk

        def read(path: Path) -> np.ndarray:
            x, _ = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
            return np.pad(x, ((0, self.chunk - len(x)), (0, 0))).T  # (2, S)

        targets = [sum(read(song / s["file"]) for s in group) for group in merged_sources(meta)]
        mix = np.sum(targets, axis=0)  # exactly the sum, as in the song
        level = np.sqrt(np.mean(mix**2)) + 1e-9
        heard = [t for t in targets if 20 * np.log10(np.sqrt(np.mean(t**2)) / level + 1e-12) > SILENT_DB]
        if self.lossy_p:
            from ..gen.augment import random_lossy

            mix = random_lossy(mix.T, self.sr, rng, self.lossy_p)[0].T
        tg = np.stack(heard) if heard else np.zeros((0, *mix.shape), np.float32)
        return torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32)), torch.from_numpy(tg.astype(np.float32))


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mixes = torch.stack([m for m, _ in batch])
    n = torch.tensor([t.shape[0] for _, t in batch])
    nmax = max(int(n.max()), 1)
    tg = torch.zeros(len(batch), nmax, *mixes.shape[1:])
    for b, (_, t) in enumerate(batch):
        tg[b, : t.shape[0]] = t
    return mixes, tg, n
