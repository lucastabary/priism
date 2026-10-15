"""Songs with one file per source, sampled as one track (k = 1) or a mix of k tracks of the same song.

A song folder holds its stems in ``sources/`` (``priism-lab gen``) or ``stems/`` (Slakh, MoisesDB copies),
or else directly (every audio file but ``mix.*``). A root is a song folder or a folder of song folders.
The kind of a stem (for the within-kind identity score) comes from the generator's ``meta.json`` when there
is one, otherwise from the file name without its leading index (``03_bass.flac`` -> ``bass``).

Mixes sum waveforms of the same song at the same time (coherent tempo and key), each with a random gain:
adding spectrograms would ignore the phase. Audio is mono and resampled to SR if needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .features import SR

AUDIO = {".flac", ".wav", ".ogg", ".aiff", ".aif"}
ACTIVE_DB = -50.0  # a 0.5 s window counts as playing above this RMS (dBFS)


@dataclass
class Song:
    path: Path
    stems: list[Path]
    kinds: list[str]
    frames: int = 0
    sr: int = SR
    _info: bool = field(default=False, repr=False)

    def info(self) -> None:
        if not self._info:
            infos = [sf.info(str(p)) for p in self.stems]
            self.frames = min(i.frames for i in infos)
            self.sr = infos[0].samplerate
            self._info = True


def _kinds(folder: Path, stems: list[Path]) -> list[str]:
    meta = folder / "meta.json"
    if meta.exists():
        by_file = {s.get("file"): s.get("kind") for s in json.loads(meta.read_text()).get("sources", [])}
        out = [by_file.get(p.relative_to(folder).as_posix()) for p in stems]
        if all(out):
            return out
    return [re.sub(r"^[\d_\-\s]+", "", p.stem).lower() or p.stem for p in stems]


def _song(folder: Path) -> Song | None:
    for sub in ("sources", "stems"):
        if (folder / sub).is_dir():
            stems = sorted(p for p in (folder / sub).iterdir() if p.suffix.lower() in AUDIO)
            break
    else:
        stems = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO and p.stem.lower() != "mix")
    return Song(folder, stems, _kinds(folder, stems)) if stems else None


def find_songs(roots: list[str], limit_frac: float = 1.0, seed: int = 0) -> list[Song]:
    songs = []
    for r in roots:
        root = Path(r)
        if any((root / f).exists() for f in ("sources", "stems", "meta.json")):  # the root is one song
            songs += [s] if (s := _song(root)) is not None else []
            continue
        songs += [s for p in sorted(root.iterdir()) if p.is_dir() and (s := _song(p)) is not None]
    if limit_frac < 1.0:  # scaling curves: a fixed random subset
        keep = np.random.default_rng(seed).permutation(len(songs))[:max(1, round(limit_frac * len(songs)))]
        songs = [songs[i] for i in sorted(keep)]
    return songs


def read_crop(song: Song, stem: int, start: int, n: int) -> np.ndarray:
    """Mono float32 crop of ``n`` samples at SR from ``start`` (in SR samples), zero-padded past the end."""
    song.info()
    if song.sr == SR:
        x, _ = sf.read(str(song.stems[stem]), start=start, stop=min(start + n, song.frames), dtype="float32",
                       always_2d=True)
    else:
        from scipy.signal import resample_poly

        a, b = int(start * song.sr / SR), int((start + n) * song.sr / SR) + 1
        x, _ = sf.read(str(song.stems[stem]), start=a, stop=min(b, song.frames), dtype="float32", always_2d=True)
        g = gcd(SR, song.sr)
        x = resample_poly(x, SR // g, song.sr // g, axis=0).astype(np.float32)
    x = x.mean(1)[:n]
    return np.pad(x, (0, n - len(x))) if len(x) < n else x


def active_frac(x: np.ndarray, win: int = SR // 2) -> float:
    n = len(x) // win
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, 1) + 1e-12)
    return float(np.mean(20 * np.log10(rms) > ACTIVE_DB))


def song_frames_sr(song: Song) -> int:
    song.info()
    return int(song.frames * SR / song.sr)


def load_mix(song: Song, k: int, rng: np.random.Generator, n: int, start: int | None = None,
             min_active: float = 0.25, gain_db: float = 6.0) -> tuple[np.ndarray, list[int]]:
    """Sum of up to ``k`` stems that play in the crop (>= ``min_active`` of its 0.5 s windows), random gains.
    Returns the mono mix and the stems used (fewer than k when the crop lacks playing stems)."""
    total = song_frames_sr(song)
    if start is None:
        start = int(rng.integers(0, max(1, total - n)))
    mix = np.zeros(n, np.float32)
    used: list[int] = []
    for s in rng.permutation(len(song.stems)):
        x = read_crop(song, int(s), start, n)
        if active_frac(x) < min_active:
            continue
        mix += x * np.float32(10 ** (rng.uniform(-gain_db, gain_db) / 20))
        used.append(int(s))
        if len(used) == k:
            break
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, used


class MixDataset(torch.utils.data.Dataset):
    """Items are (song index, k, seed) tuples from ``KSampler``; returns a fixed-length mono crop."""

    def __init__(self, songs: list[Song], n_samples: int):
        self.songs = songs
        self.n = n_samples

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, item):
        i, k, seed = item
        rng = np.random.default_rng(seed)
        wave, used = load_mix(self.songs[i], k, rng, self.n)
        return {"wave": torch.from_numpy(wave), "k": len(used)}


def parse_schedule(spec: str) -> list[tuple[int, int]]:
    """'0:1,20000:2,30000:4' -> [(0, 1), (20000, 2), (30000, 4)]: the largest k allowed from each step on."""
    out = sorted((int(a), int(b)) for a, b in (p.split(":") for p in spec.split(",")))
    if not out or out[0][0] != 0:
        raise ValueError(f"k schedule must start at step 0: {spec!r}")
    return out


class KSampler(torch.utils.data.Sampler):
    """Endless (song, k, seed) draws. The training loop sets ``step``; once k may exceed 1, a ``mono_frac``
    share of draws stays single-track (replay against forgetting), the rest take k uniform in 2..kmax."""

    def __init__(self, n_songs: int, schedule: list[tuple[int, int]], mono_frac: float = 0.5, seed: int = 0):
        self.n = n_songs
        self.schedule = schedule
        self.mono_frac = mono_frac
        self.rng = np.random.default_rng(seed)
        self.step = 0

    def kmax(self) -> int:
        return max(k for s, k in self.schedule if s <= self.step)

    def __iter__(self):
        while True:
            kmax = self.kmax()
            k = 1 if kmax <= 1 or self.rng.random() < self.mono_frac else int(self.rng.integers(2, kmax + 1))
            yield int(self.rng.integers(self.n)), k, int(self.rng.integers(2**31))
