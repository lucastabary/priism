"""Separate a whole song with a model that only sees short windows.

The separator works on excerpts of a few seconds and its 16 slots carry no fixed
meaning: the kick can be slot 3 in one window and slot 9 in the next. A whole song
is therefore cut into overlapping windows, and the tracks of each window are linked
to those of the previous one where the two overlap (same audio there, so the same
source), with the Hungarian algorithm. Tracks are crossfaded over the overlaps.

A source that stops and comes back later (a breakdown) starts a new track; a second
pass joins two tracks that never sound at the same time and whose identity vectors
(attractors) agree. The rest is mix minus the tracks, so the sum is exactly the mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

# model_fn(window (C, W)) -> sources (K, C, W), existence probabilities (K,), attractors (K, D)
ModelFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class _Track:
    audio: np.ndarray  # (C, S) crossfaded so far
    attractors: list[np.ndarray] = field(default_factory=list)
    windows: list[int] = field(default_factory=list)
    last: np.ndarray | None = None  # its audio in the last window it was in (C, W)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _ramps(win: int, overlap: int, first: bool, last: bool) -> np.ndarray:
    """Window weights: linear fades over the overlaps, so consecutive windows sum to 1."""
    w = np.ones(win, dtype=np.float32)
    if overlap:
        ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        if not first:
            w[:overlap] = ramp
        if not last:
            w[-overlap:] = ramp[::-1]
    return w


def split_song(model_fn: ModelFn, mix: np.ndarray, sr: int, window_s: float = 8.0, overlap_s: float = 2.0,
               threshold: float = 0.5, link_min: float = 0.3, join_min: float = 0.85,
               silent_db: float = -50.0) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """mix (C, S) -> tracks [(C, S)] loudest first, rest (C, S), and per-track info.

    ``link_min``: lowest similarity to continue a track into the next window.
    ``join_min``: lowest attractor similarity to join two tracks that never overlap (None to skip).
    """
    C, S = mix.shape
    win = int(window_s * sr)
    overlap = min(int(overlap_s * sr), win // 2)
    hop = win - overlap
    starts = list(range(0, max(S - overlap, 1), hop)) if S > win else [0]
    mix_power = float(np.mean(mix ** 2)) + 1e-12
    tracks: list[_Track] = []
    prev: list[int] = []  # track index of each kept slot of the previous window
    for wi, st in enumerate(starts):
        chunk = mix[:, st:st + win]
        n = chunk.shape[1]
        if n < win:
            chunk = np.pad(chunk, ((0, 0), (0, win - n)))
        src, prob, att = model_fn(chunk)
        src = src[:, :, :n]
        loud = np.mean(src ** 2, axis=(1, 2)) > mix_power * 10 ** (silent_db / 10)
        keep = np.flatnonzero((prob > threshold) & loud)
        # Link to the tracks of the previous window on the audio both windows share.
        assigned = {}
        if prev and keep.size and wi > 0:
            sim = np.zeros((len(prev), keep.size))
            for a, t in enumerate(prev):
                old = tracks[t].last[:, hop:hop + overlap]
                for b, k in enumerate(keep):
                    new = src[k][:, :overlap]
                    e_old, e_new = np.mean(old ** 2), np.mean(new ** 2)
                    floor = mix_power * 10 ** (silent_db / 10)
                    att_sim = _cos(np.mean(tracks[t].attractors, 0), att[k])
                    if e_old > floor and e_new > floor:
                        sim[a, b] = 0.7 * _cos(old, new) + 0.3 * att_sim
                    else:  # nothing to hear in the overlap: only the identity vectors can tell
                        sim[a, b] = att_sim
            rows, cols = linear_sum_assignment(-sim)
            assigned = {int(keep[c]): prev[r] for r, c in zip(rows, cols) if sim[r, c] >= link_min}
        weights = _ramps(win, overlap, wi == 0, wi == len(starts) - 1)[:n]
        now = []
        for k in keep:
            k = int(k)
            if k in assigned:
                t = assigned[k]
            else:
                tracks.append(_Track(np.zeros((C, S), dtype=np.float32)))
                t = len(tracks) - 1
            tr = tracks[t]
            tr.audio[:, st:st + n] += src[k] * weights
            tr.attractors.append(att[k])
            tr.windows.append(wi)
            tr.last = np.pad(src[k], ((0, 0), (0, win - n)))
            now.append(t)
        prev = now

    # Second pass: a source that paused and came back is one track if it never overlaps itself.
    if join_min is not None:
        merged = True
        while merged:
            merged = False
            best, pair = join_min, None
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    if set(tracks[i].windows) & set(tracks[j].windows):
                        continue
                    s = _cos(np.mean(tracks[i].attractors, 0), np.mean(tracks[j].attractors, 0))
                    if s >= best:
                        best, pair = s, (i, j)
            if pair:
                i, j = pair
                tracks[i].audio += tracks[j].audio
                tracks[i].attractors += tracks[j].attractors
                tracks[i].windows = sorted(tracks[i].windows + tracks[j].windows)
                del tracks[j]
                merged = True

    tracks.sort(key=lambda t: -float(np.mean(t.audio ** 2)))
    audio = [t.audio for t in tracks]
    rest = mix - (np.sum(audio, axis=0) if audio else 0.0)
    info = [{"windows": len(t.windows), "first_s": round(starts[t.windows[0]] / sr, 2),
             "last_s": round(min(starts[t.windows[-1]] + win, S) / sr, 2),
             "rms_db": round(10 * np.log10(np.mean(t.audio ** 2) / mix_power + 1e-12), 1)} for t in tracks]
    return audio, rest, info


def torch_model_fn(model, device: str = "cpu") -> ModelFn:
    """Wrap a separator from ``load_run`` as a ``ModelFn``."""
    import torch

    @torch.no_grad()
    def fn(chunk: np.ndarray):
        out = model(torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None].to(device))
        return (out["sources"][0].float().cpu().numpy(), torch.sigmoid(out["exist_logits"][0].float()).cpu().numpy(),
                out["attractors"][0].float().cpu().numpy())

    return fn


def split_file(run, path, out, window_s: float = 8.0, overlap_s: float = 2.0, threshold: float = 0.5,
               device: str = "cpu", msst_path=None, start_s: float = 0.0, duration_s: float | None = None,
               log=print) -> list[dict]:
    """Separate one audio file into ``out/mix.wav``, ``out/pistes/NN_piste.wav``, ``out/pistes/reste.wav``."""
    import json
    import subprocess
    from pathlib import Path

    import soundfile as sf

    from .train import load_run

    model, sr = load_run(run, device, msst_path)
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start_s), "-i", str(path)]
    if duration_s:
        cmd += ["-t", str(duration_s)]
    raw = subprocess.run(cmd + ["-f", "f32le", "-ac", "2", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    mix = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    tracks, rest, info = split_song(torch_model_fn(model, device), mix, sr, window_s, overlap_s, threshold)
    out = Path(out)
    (out / "pistes").mkdir(parents=True, exist_ok=True)
    sf.write(out / "mix.wav", mix.T, sr, subtype="FLOAT")
    for i, t in enumerate(tracks):
        sf.write(out / "pistes" / f"{i:02d}_piste.wav", t.T, sr, subtype="FLOAT")
    sf.write(out / "pistes" / "reste.wav", rest.T, sr, subtype="FLOAT")
    (out / "pistes.json").write_text(json.dumps({"source": str(path), "tracks": info}, indent=1))
    log(f"{path}: {len(tracks)} tracks")
    return info
