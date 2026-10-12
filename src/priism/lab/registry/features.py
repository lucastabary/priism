"""Cache of frozen-trunk features: the expensive part (the 175 M-parameter BS-Roformer-SW trunk) runs once
per song, everything after it (tracking methods, the identity head) trains on the cache, even on a CPU.

Per song, ``<name>.npz`` holds:

- ``tokens`` (T, Nb, k) float16: the trunk's final features per frame and band, pooled in time by ``pool``
  and reduced from D to k numbers by a PCA fitted on training songs only (``Projection``);
- ``energy`` (J, T, Nb) float16: log10 power of each reference source in the same tokens, so every
  metric knows who owns which token without storing audio;
- ``labels``: JSON list, one dict per source (``name``, ``instrument`` id shared by twins, ``entry_s``,
  ``fx_sections`` when the generator automated its effects); ``frame_s``: seconds per token frame.

The trunk sees the song through overlapping windows (``window_s``) and only the centre of each window
is kept (``margin_s`` dropped at both inner edges), so no token comes from the edge of a window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import torch


@dataclass
class Song:
    """One reference song: mix (C, S), sources (J, C, S) that sum to it, one label dict per source."""
    name: str
    mix: np.ndarray
    sources: np.ndarray
    labels: list[dict]


class Trunk(Protocol):
    sample_rate: int
    hop: int
    n_fft: int
    band_bins: np.ndarray  # (Nb, 2) [lo, hi) STFT bins of each band

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:  # (C, S) -> (T, Nb, D), T = S // hop + 1
        ...


class MsstTrunk:
    """The BS-Roformer trunk up to its final norm (the separator's ``_features``), frozen, in bf16 on GPU."""

    def __init__(self, config: str | Path, ckpt: str | Path, msst_path: str | Path, device: str = "cuda"):
        from ..model.msst_core import MsstAttractorSeparator, load_msst_roformer

        import yaml

        cfg = yaml.load(Path(config).read_text(), Loader=yaml.FullLoader)
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", 44100))
        self.model = MsstAttractorSeparator(load_msst_roformer(config, ckpt, msst_path), max_sources=1,
                                            sample_rate=self.sample_rate).eval().requires_grad_(False).to(device)
        r = self.model.r
        self.hop, self.n_fft = int(r.stft_kwargs["hop_length"]), int(r.stft_kwargs["n_fft"])
        edges = (self.model._band_hz(self.sample_rate) * self.n_fft / self.sample_rate).round().long().numpy()
        self.band_bins = edges
        self.device = device

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.startswith("cuda")):
            h, _, _ = self.model._features(wav[None].to(self.device))
        return h[0].float().cpu()


def song_tokens(trunk: Trunk, wav: np.ndarray, window_s: float = 8.0, margin_s: float = 1.0) -> np.ndarray:
    """(C, S) -> (T, Nb, D) trunk features for the whole song, T = S // hop + 1 (centred STFT frames)."""
    hop = trunk.hop
    S = wav.shape[1]
    n_frames = S // hop + 1
    win_f = int(round(window_s * trunk.sample_rate / hop))
    m = int(round(margin_s * trunk.sample_rate / hop))
    step_f = win_f - 2 * m
    if step_f <= 0:
        raise ValueError("window too short for its margins")
    out, g, k = [], 0, 0
    while g < n_frames:
        start_f = k * step_f
        seg = wav[:, start_f * hop: start_f * hop + win_f * hop]
        if seg.shape[1] < win_f * hop:  # the last window: zeros after the song, its frames are dropped below
            seg = np.pad(seg, ((0, 0), (0, win_f * hop - seg.shape[1])))
        h = trunk(torch.from_numpy(np.ascontiguousarray(seg, np.float32)))
        last = start_f + h.shape[0] >= n_frames
        end = n_frames - start_f if last else m + step_f
        out.append(h[g - start_f: end].numpy())
        g, k = start_f + end, k + 1
    return np.concatenate(out)[:n_frames]


def pool_time(x: np.ndarray, pool: int, axis: int = 0, reduce: str = "mean") -> np.ndarray:
    """Average (or sum) groups of ``pool`` frames along ``axis``; the last group may be shorter."""
    if pool == 1:
        return x
    x = np.moveaxis(x, axis, 0)
    n = -(-x.shape[0] // pool)
    pad = n * pool - x.shape[0]
    xs = np.concatenate([x, np.zeros((pad, *x.shape[1:]), x.dtype)]) if pad else x
    s = xs.reshape(n, pool, *x.shape[1:]).sum(1)
    if reduce == "mean":
        counts = np.full(n, pool, np.float64)
        counts[-1] = pool - pad
        s = s / counts.reshape(-1, *[1] * (x.ndim - 1))
    return np.moveaxis(s.astype(x.dtype), 0, axis)


def band_energy(trunk: Trunk, wav: np.ndarray) -> np.ndarray:
    """(C, S) -> (T, Nb) power of one signal in each trunk band and frame (same STFT frames as the trunk)."""
    dev = getattr(trunk, "device", "cpu")  # on the GPU with a GPU trunk: songs with many tracks stay fast
    x = torch.from_numpy(np.ascontiguousarray(wav, np.float32)).to(dev)
    spec = torch.stft(x, trunk.n_fft, trunk.hop, window=torch.hann_window(trunk.n_fft, device=dev), center=True,
                      return_complex=True)
    p = spec.abs().pow(2).sum(0).double()  # (F, T), channels summed; float64 cumsum keeps quiet bands exact
    c = torch.cat([torch.zeros(1, p.shape[1], dtype=p.dtype, device=dev), p.cumsum(0)])
    lo = torch.from_numpy(trunk.band_bins[:, 0]).to(dev).clamp(max=p.shape[0])
    hi = torch.from_numpy(trunk.band_bins[:, 1]).to(dev).clamp(max=p.shape[0])
    return (c[hi] - c[lo]).T.float().cpu().numpy()


class Projection:
    """PCA of trunk tokens (D -> k), fitted on training songs only so no test song shapes the features."""

    def __init__(self, mean: np.ndarray | None = None, basis: np.ndarray | None = None):
        self.mean, self.basis = mean, basis
        self._n, self._s, self._ss = 0, None, None

    def update(self, h: np.ndarray, weight: np.ndarray | None = None) -> None:
        """Accumulate tokens (..., D); ``weight`` (...,) drops silent tokens (weight 0)."""
        x = h.reshape(-1, h.shape[-1]).astype(np.float64)
        if weight is not None:
            x = x[weight.reshape(-1) > 0]
        if self._s is None:
            self._s, self._ss = np.zeros(x.shape[1]), np.zeros((x.shape[1], x.shape[1]))
        self._n += len(x)
        self._s += x.sum(0)
        self._ss += x.T @ x

    def fit(self, k: int) -> Projection:
        mean = self._s / self._n
        cov = self._ss / self._n - np.outer(mean, mean)
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1][:k]
        self.mean, self.basis = mean.astype(np.float32), v[:, order].astype(np.float32)
        self.explained = float(w[order].sum() / w.sum())
        return self

    def __call__(self, h: np.ndarray) -> np.ndarray:
        return (h - self.mean) @ self.basis

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, basis=self.basis, explained=getattr(self, "explained", np.nan))

    @classmethod
    def load(cls, path: str | Path) -> Projection:
        d = np.load(path)
        p = cls(d["mean"], d["basis"])
        p.explained = float(d["explained"])
        return p


def song_cache(trunk: Trunk, song: Song, proj: Projection, pool: int = 4, window_s: float = 8.0,
               margin_s: float = 1.0) -> dict:
    """Everything stored for one song (see the module docstring)."""
    h = song_tokens(trunk, song.mix, window_s, margin_s)
    tokens = pool_time(proj(h), pool)
    energy = np.stack([pool_time(band_energy(trunk, s), pool, reduce="sum") for s in song.sources])
    return {"tokens": tokens.astype(np.float16), "energy": np.log10(energy + 1e-10).astype(np.float16),
            "labels": json.dumps(song.labels), "frame_s": trunk.hop * pool / trunk.sample_rate}


def write_cache(path: str | Path, cache: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **cache)
    tmp.replace(path)  # a half-written file never looks finished


@dataclass
class Cached:
    """One cached song, kept small in memory (a training run holds hundreds): tokens as stored (float16),
    energy as stored (log10, float16). ``energy`` gives linear float32 power on demand, ``energy_at`` a time
    slice of it, so a training step only expands the chunks it samples."""
    name: str
    tokens: np.ndarray  # (T, Nb, k) float16
    log_energy: np.ndarray  # (J, T, Nb) log10 power, float16
    labels: list[dict]
    frame_s: float
    path: str = ""  # the .npz it came from (parallel evaluation reloads it in each worker)

    @property
    def energy(self) -> np.ndarray:  # (J, T, Nb) linear power, float32
        return 10 ** self.log_energy.astype(np.float32)

    def energy_at(self, t0: int, t1: int) -> np.ndarray:
        return 10 ** self.log_energy[:, t0:t1].astype(np.float32)

    @property
    def total_max(self) -> float:
        """Loudest token of the song (all sources summed): the reference of the audibility floor."""
        if not hasattr(self, "_tmax"):
            self._tmax = float(max(self.energy_at(t, t + 512).sum(0).max() for t in range(0, self.tokens.shape[0], 512)))
        return self._tmax

    @property
    def instruments(self) -> np.ndarray:
        """(J,) instrument id of each source: twins and ambiguous pairs share one."""
        return np.array([lab["instrument"] for lab in self.labels])


def load_cache(path: str | Path, lazy: bool = False) -> Cached:
    """``lazy``: labels and frame rate only (no arrays decompressed), for a process that only hands paths to
    workers; the arrays are then loaded by ``load_cache(song.path)`` where needed."""
    d = np.load(path)
    if lazy:
        empty = np.zeros((0, 0, 0), np.float16)
        return Cached(Path(path).stem, empty, empty, json.loads(str(d["labels"])), float(d["frame_s"]), str(path))
    return Cached(Path(path).stem, d["tokens"].astype(np.float16), d["energy"].astype(np.float16),
                  json.loads(str(d["labels"])), float(d["frame_s"]), str(path))


def load_set(folder: str | Path, lazy: bool = False) -> list[Cached]:
    return [load_cache(p, lazy) for p in sorted(Path(folder).glob("*.npz")) if not p.name.endswith(".tmp.npz")]


# ------------------------------------------------------------------------------------------- song readers

def _read(path: Path, sr: int) -> np.ndarray:
    x, fsr = sf.read(path, dtype="float32", always_2d=True)
    if fsr != sr:
        raise ValueError(f"{path}: {fsr} Hz, expected {sr} (resample the set first)")
    return np.repeat(x.T, 2, 0) if x.shape[1] == 1 else x.T[:2]


def read_synth(folder: str | Path, sr: int) -> Song:
    """A song from ``priism-lab gen``. Twins and ambiguous pairs share an instrument id (their original's)."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    srcs = [_read(folder / s["file"], sr) for s in meta["sources"]]
    bar_s = 4 * 60.0 / meta["bpm"]
    labels = []
    for s in meta["sources"]:
        lab = {"name": f"{s['id']:02d}_{s['kind']}", "kind": s["kind"],
               "instrument": s["twin_of"] if s["twin_of"] is not None else s["id"],
               "entry_s": min(s["bars"]) * bar_s if s["bars"] else 0.0}
        if "fx_sections" in s:
            lab["fx_sections"] = s["fx_sections"]
        labels.append(lab)
    sources = np.stack(srcs)
    return Song(folder.name, sources.sum(0), sources, labels)


def read_ni(path: str | Path, sr: int) -> Song:
    """A NI .stem.mp4: 4 grouped stems (drums, bass, synths, vocals...), each its own instrument."""
    from ..model.evaluate import read_stem_mp4

    mix, stems = read_stem_mp4(path, sr)
    labels = [{"name": f"stem{j}", "kind": f"stem{j}", "instrument": j, "entry_s": None} for j in range(len(stems))]
    return Song(Path(path).name.removesuffix(".stem.mp4"), mix, stems, labels)


def read_multitrack(folder: str | Path, sr: int) -> Song:
    """A multitrack folder with one audio file per track, possibly in one subfolder per stem type
    (MoisesDB: ``<song>/<stem>/<track id>.flac`` plus ``data.json``). Every file is its own source and its own
    instrument (fine level). With a MoisesDB ``data.json``, each label gets the track type as ``kind`` (e.g.
    bass_guitar, cymbals), the stem name as ``stem`` (drums, guitar...), ``has_bleed``, and ``group_key`` =
    the artist, so splits keep an artist on one side; otherwise ``kind`` is the subfolder name."""
    import json

    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".wav", ".flac")
                   and p.stem.lower() not in ("mix", "mixture"))
    meta = json.loads((folder / "data.json").read_text()) if (folder / "data.json").exists() else {}
    tracks = {t["id"]: (st.get("stemName"), t) for st in meta.get("stems", []) for t in st.get("tracks", [])}
    srcs = [_read(p, sr) for p in files]
    n = min(s.shape[1] for s in srcs)
    sources = np.stack([s[:, :n] for s in srcs])
    labels = []
    for j, p in enumerate(files):
        lab = {"name": str(p.relative_to(folder).with_suffix("")).replace("\\", "/"),
               "kind": p.parent.name if p.parent != folder else p.stem, "instrument": j, "entry_s": None}
        if p.stem in tracks:
            stem, t = tracks[p.stem]
            lab.update(kind=t.get("trackType", lab["kind"]), stem=stem, has_bleed=bool(t.get("has_bleed")))
        if meta:
            lab.update(group_key=meta.get("artist") or folder.name, genre=meta.get("genre"))
        labels.append(lab)
    return Song(folder.name, sources.sum(0), sources, labels)
