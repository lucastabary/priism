"""Songs with one file per source, sampled as one track (k = 1) or a mix of k tracks of the same song.

A song folder holds its stems in ``sources/`` (``priism-lab gen``) or ``stems/`` (Slakh, MoisesDB copies),
or else directly (every audio file but ``mix.*``). A root is a song folder or a folder of song folders.
The kind of a stem (for the within-kind identity score) comes from the generator's ``meta.json`` when there
is one, otherwise from the file name without its leading index (``03_bass.flac`` -> ``bass``).

Mixes sum waveforms of the same song at the same time (coherent tempo and key), each with a random gain:
adding spectrograms would ignore the phase. Audio is mono and resampled to SR if needed.

Paired FX views (docs/cleave.md 2.9): one k = 1 crop rendered through two random pedalboard chains, so the
model can learn that what the chains change belongs to V-bar, not to Z. With ``other_view``, a paired crop also
brings another stem of the same song, same time, through chain A: V-bar read from it can describe the
chain but not the timbre of the stem being predicted (V-bar from the stem's own neighbours carried its
identity, so the predictor stopped needing Z).
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
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
        if not stems:  # MoisesDB layout: one subfolder per stem class, the class is the kind
            stems = sorted(p for d in folder.iterdir() if d.is_dir() for p in d.iterdir() if p.suffix.lower() in AUDIO)
            return Song(folder, stems, [p.parent.name.lower() for p in stems]) if stems else None
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


# Paired FX views (docs/cleave.md 2.9). A chain is a fixed-length vector of uniform draws in [0, 1], one per
# column, 0 when its effect is off: the target of the Z -> FX leakage probe, scale-free for R². Every effect
# here belongs to V (distortion and filters included): whatever we randomise is, by definition, not identity.
FX_COLS = ["gain", "hp_on", "hp_hz", "lp_on", "lp_hz", "eq_on", "eq_hz", "eq_db", "comp_on", "comp_thr",
           "comp_ratio", "dist_on", "dist_drive", "chorus_on", "chorus_rate", "chorus_depth", "phaser_on",
           "phaser_rate", "phaser_depth", "delay_on", "delay_s", "delay_fb", "delay_mix", "rev_on", "rev_size",
           "rev_wet"]
FX_P = 0.5  # each effect is on with this probability; chorus and phaser share one modulation slot


def sample_fx(rng: np.random.Generator) -> np.ndarray:
    """A random chain as a (len(FX_COLS),) float32 vector."""
    v = dict.fromkeys(FX_COLS, 0.0)
    v["gain"] = rng.random()
    groups = {"hp": ["hp_hz"], "lp": ["lp_hz"], "eq": ["eq_hz", "eq_db"], "comp": ["comp_thr", "comp_ratio"],
              "dist": ["dist_drive"], "delay": ["delay_s", "delay_fb", "delay_mix"], "rev": ["rev_size", "rev_wet"]}
    for g, cols in groups.items():
        if rng.random() < FX_P:
            v[g + "_on"] = 1.0
            for c in cols:
                v[c] = rng.random()
    if rng.random() < FX_P:
        g = "chorus" if rng.random() < 0.5 else "phaser"
        v[g + "_on"], v[g + "_rate"], v[g + "_depth"] = 1.0, rng.random(), rng.random()
    return np.array([v[c] for c in FX_COLS], np.float32)


def render_fx(x: np.ndarray, fx: np.ndarray, sr: int = SR) -> np.ndarray:
    """Mono ``x`` through the chain ``fx`` (fixed studio order), same length, peak kept under 0.99."""
    import pedalboard as pb

    v = dict(zip(FX_COLS, fx.tolist()))

    def lin(u, lo, hi):
        return lo + u * (hi - lo)

    def log(u, lo, hi):
        return lo * (hi / lo) ** u

    plugins = []
    if v["hp_on"]:
        plugins.append(pb.HighpassFilter(log(v["hp_hz"], 40.0, 800.0)))
    if v["lp_on"]:
        plugins.append(pb.LowpassFilter(log(v["lp_hz"], 1500.0, 16000.0)))
    if v["eq_on"]:
        plugins.append(pb.PeakFilter(log(v["eq_hz"], 200.0, 6000.0), lin(v["eq_db"], -9.0, 9.0), 1.0))
    if v["comp_on"]:
        plugins.append(pb.Compressor(lin(v["comp_thr"], -40.0, -10.0), lin(v["comp_ratio"], 2.0, 10.0)))
    if v["dist_on"]:
        plugins.append(pb.Distortion(lin(v["dist_drive"], 5.0, 35.0)))
    if v["chorus_on"]:
        plugins.append(pb.Chorus(rate_hz=lin(v["chorus_rate"], 0.2, 3.0), depth=lin(v["chorus_depth"], 0.1, 0.9),
                                 mix=0.5))
    if v["phaser_on"]:
        plugins.append(pb.Phaser(rate_hz=lin(v["phaser_rate"], 0.1, 2.0), depth=lin(v["phaser_depth"], 0.3, 1.0),
                                 mix=0.5))
    if v["delay_on"]:
        plugins.append(pb.Delay(lin(v["delay_s"], 0.08, 0.6), lin(v["delay_fb"], 0.0, 0.6),
                                lin(v["delay_mix"], 0.15, 0.5)))
    if v["rev_on"]:
        plugins.append(pb.Reverb(room_size=lin(v["rev_size"], 0.2, 0.95), wet_level=lin(v["rev_wet"], 0.1, 0.6),
                                 dry_level=0.8))
    plugins.append(pb.Gain(lin(v["gain"], -9.0, 3.0)))
    y = pb.Pedalboard(plugins)(x[None].astype(np.float32), sr)[0][: len(x)]
    y = np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y
    peak = float(np.abs(y).max())
    return (y * (0.99 / peak) if peak > 0.99 else y).astype(np.float32)


def paired_views(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The same crop through two independent random chains A and B: (wave A, wave B, fx A, fx B)."""
    fa, fb = sample_fx(rng), sample_fx(rng)
    return render_fx(x, fa), render_fx(x, fb), fa, fb


def other_stem(song: Song, used: list[int], start: int, n: int, rng: np.random.Generator,
               min_active: float = 0.25) -> np.ndarray | None:
    """A random other stem of ``song`` playing in the same crop, or None when none plays."""
    for s in rng.permutation(len(song.stems)):
        if int(s) in used:
            continue
        x = read_crop(song, int(s), start, n)
        if active_frac(x) >= min_active:
            return x
    return None


# RAM song cache. On Chav the data sits on a USB drive read through WSL: random crop reads (plus the silent
# stems load_mix reads before finding a playing one) bound the step time, not the CPU. A cached song is read
# once, whole and sequentially, kept mono float16, with an activity index (playing 0.5 s windows per stem)
# that picks crops where k stems play without reading anything. ``PoolBatchSampler`` keeps each worker on a
# few songs, so the cache stays small and each song serves many crops.

WIN = SR // 2  # activity window, as in active_frac


@dataclass
class Loaded:
    stems: np.ndarray  # (n_stems, frames) float16 mono at SR
    active: np.ndarray  # (n_stems, frames // WIN) bool: the window's RMS is above ACTIVE_DB

    @property
    def nbytes(self) -> int:
        return self.stems.nbytes + self.active.nbytes


def song_bytes(song: Song) -> int:
    """RAM of ``load_song(song)``, from the file headers only."""
    return song_frames_sr(song) * len(song.stems) * 2


def load_song(song: Song) -> Loaded:
    song.info()
    xs = []
    for p in song.stems:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly

            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        xs.append(x)
    n = min(len(x) for x in xs)
    w = n // WIN
    active = np.stack([20 * np.log10(np.sqrt(np.mean(x[: w * WIN].reshape(w, WIN) ** 2, 1) + 1e-12)) > ACTIVE_DB
                       for x in xs])
    return Loaded(np.stack([x[:n].astype(np.float16) for x in xs]), active)


def _cut(loaded: Loaded, s: int, start: int, n: int) -> np.ndarray:
    x = loaded.stems[s, start:start + n].astype(np.float32)
    return np.pad(x, (0, n - len(x))) if len(x) < n else x


def _crop_frac(loaded: Loaded, start: int, n: int) -> np.ndarray:
    """Playing share of each stem over the windows of the crop (as active_frac on a window-aligned crop)."""
    a = loaded.active[:, start // WIN:start // WIN + max(1, n // WIN)]
    return a.mean(1) if a.shape[1] else np.zeros(len(a))


def pool_mix(loaded: Loaded, k: int, rng: np.random.Generator, n: int, min_active: float = 0.25,
             gain_db: float = 6.0) -> tuple[np.ndarray, list[int], int]:
    """``load_mix`` on a cached song, without reads: the crop starts (on the 0.5 s grid) where at least k stems
    play, or where the most stems play when no start has k. Returns the mix, the stems used and the start."""
    S, W = loaded.active.shape
    nw = max(1, n // WIN)
    if W >= nw:
        c = np.concatenate([np.zeros((S, 1)), np.cumsum(loaded.active, 1)], 1)
        n_ok = ((c[:, nw:] - c[:, :-nw]) / nw >= min_active).sum(0)  # playing stems per window start
        good = np.flatnonzero(n_ok >= min(k, n_ok.max()))
        start = int(rng.choice(good)) * WIN
    else:
        start = 0
    cand = np.flatnonzero(_crop_frac(loaded, start, n) >= min_active)
    mix = np.zeros(n, np.float32)
    used = [int(s) for s in rng.permutation(cand)[:k]]
    for s in used:
        mix += _cut(loaded, s, start, n) * np.float32(10 ** (rng.uniform(-gain_db, gain_db) / 20))
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, used, start


def pool_other(loaded: Loaded, used: list[int], start: int, n: int, rng: np.random.Generator,
               min_active: float = 0.25) -> np.ndarray | None:
    """``other_stem`` on a cached song."""
    frac = _crop_frac(loaded, start, n)
    cand = [s for s in np.flatnonzero(frac >= min_active) if int(s) not in used]
    return _cut(loaded, int(rng.choice(cand)), start, n) if cand else None


class MixDataset(torch.utils.data.Dataset):
    """Items are (song index, k, seed) tuples from ``KSampler``; returns a fixed-length mono crop.

    A ``pair_frac`` share of the k = 1 crops come as paired FX views: ``wave`` is view A, ``wave_b`` view B,
    with their chains. With ``other_view``, they also carry ``wave_o``, another playing stem through chain A
    (a crop with no other playing stem then stays unpaired). Batch them with ``collate``, which stacks the
    paired ones apart.

    With ``cache_songs`` > 0, each process keeps that many whole songs in RAM (least recently used out) and
    crops them with ``pool_mix``: pair it with ``PoolBatchSampler`` so a worker sees few distinct songs."""

    def __init__(self, songs: list[Song], n_samples: int, pair_frac: float = 0.0, other_view: bool = False,
                 cache_songs: int = 0):
        self.songs = songs
        self.n = n_samples
        self.pair_frac = pair_frac
        self.other_view = other_view
        self.cache_songs = cache_songs
        self._cache: OrderedDict[int, Loaded] = OrderedDict()

    def __len__(self) -> int:
        return len(self.songs)

    def loaded(self, i: int) -> Loaded:
        if i in self._cache:
            self._cache.move_to_end(i)
        else:
            while len(self._cache) >= self.cache_songs:
                self._cache.popitem(last=False)
            self._cache[i] = load_song(self.songs[i])
        return self._cache[i]

    def __getitem__(self, item):
        i, k, seed = item
        rng = np.random.default_rng(seed)
        song = self.songs[i]
        if self.cache_songs:
            loaded = self.loaded(i)
            wave, used, start = pool_mix(loaded, k, rng, self.n)
        else:
            loaded = None
            start = int(rng.integers(0, max(1, song_frames_sr(song) - self.n)))  # load_mix's own first draw
            wave, used = load_mix(song, k, rng, self.n, start)
        if len(used) == 1 and self.pair_frac > 0 and rng.random() < self.pair_frac:
            if not self.other_view:
                other = None
            elif loaded is not None:
                other = pool_other(loaded, used, start, self.n, rng)
            else:
                other = other_stem(song, used, start, self.n, rng)
            if not self.other_view or other is not None:
                a, b, fa, fb = paired_views(wave, rng)
                item = {"wave": torch.from_numpy(a), "k": 1, "wave_b": torch.from_numpy(b),
                        "fx_a": torch.from_numpy(fa), "fx_b": torch.from_numpy(fb)}
                if other is not None:
                    item["wave_o"] = torch.from_numpy(render_fx(other, fa))
                return item
        return {"wave": torch.from_numpy(wave), "k": len(used)}


def collate(items: list[dict]) -> dict:
    """Stack ``wave`` for all items; ``wave_b`` / ``fx_*`` (and ``wave_o`` when they have it) only for the
    paired ones, whose batch positions are ``pair_idx`` (empty tensors when none is paired: unpaired items
    carry no dummy second view)."""
    paired = [j for j, it in enumerate(items) if "wave_b" in it]
    n, d = items[0]["wave"].shape[0], len(FX_COLS)
    out = {"wave": torch.stack([it["wave"] for it in items]), "k": torch.tensor([it["k"] for it in items]),
           "pair_idx": torch.tensor(paired, dtype=torch.long)}
    for key, shape in (("wave_b", (0, n)), ("fx_a", (0, d)), ("fx_b", (0, d))):
        out[key] = torch.stack([items[j][key] for j in paired]) if paired else torch.zeros(shape)
    if paired and all("wave_o" in items[j] for j in paired):
        out["wave_o"] = torch.stack([items[j]["wave_o"] for j in paired])
    return out


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

    def draw_k(self) -> int:
        kmax = self.kmax()
        return 1 if kmax <= 1 or self.rng.random() < self.mono_frac else int(self.rng.integers(2, kmax + 1))

    def __iter__(self):
        while True:
            k = self.draw_k()
            yield int(self.rng.integers(self.n)), k, int(self.rng.integers(2**31))


class PoolBatchSampler(KSampler):
    """Endless batches of (song, k, seed) items for a ``MixDataset`` with a song cache.

    The DataLoader hands batch i to worker i % workers (round robin over its live workers), so batch i draws
    from pool i % pools: each worker only ever needs the songs of its own pool in RAM. A song leaves its pool
    after ``reuse`` crops, replaced by a random song that is in no pool when there is one. If the round robin
    ever differs, a worker just loads a song it lacks: slower, never wrong. Same k schedule as ``KSampler``."""

    def __init__(self, n_songs: int, batch: int, schedule: list[tuple[int, int]], pools: int, pool_songs: int,
                 reuse: int, mono_frac: float = 0.5, seed: int = 0):
        super().__init__(n_songs, schedule, mono_frac, seed)
        self.batch = batch
        self.reuse = reuse
        order = self.rng.permutation(n_songs)
        self.pools = [[int(order[(p * pool_songs + j) % n_songs]) for j in range(pool_songs)] for p in range(pools)]
        self.left = [[reuse] * pool_songs for _ in range(pools)]

    def _replace(self, p: int, j: int) -> None:
        busy = {s for pool in self.pools for s in pool}
        free = [s for s in range(self.n) if s not in busy]
        self.pools[p][j] = int(self.rng.choice(free)) if free else int(self.rng.integers(self.n))
        self.left[p][j] = self.reuse

    def __iter__(self):
        i = 0
        while True:
            p = i % len(self.pools)
            items = []
            for _ in range(self.batch):
                j = int(self.rng.integers(len(self.pools[p])))
                items.append((self.pools[p][j], self.draw_k(), int(self.rng.integers(2**31))))
                self.left[p][j] -= 1
                if self.left[p][j] <= 0:
                    self._replace(p, j)
            yield items
            i += 1
