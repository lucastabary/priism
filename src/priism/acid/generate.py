"""Render synthetic acid lines to disk, one WAV and one JSON per line."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from .fx import apply_fx
from .params import AcidParams, sample_params
from .synth import render_dry


def render(p: AcidParams) -> np.ndarray:
    """Stereo float32 acid line, normalised to ``p.peak_dbfs``."""
    out = apply_fx(render_dry(p), p)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out * (10 ** (p.peak_dbfs / 20) / peak)
    return out.astype(np.float32)


def _render_one(args: tuple[int, float, int, str]) -> str:
    seed, duration_s, sample_rate, out_dir = args
    p = sample_params(seed, duration_s=duration_s, sample_rate=sample_rate)
    stem = Path(out_dir) / f"acid_{seed:07d}"
    sf.write(stem.with_suffix(".wav"), render(p), sample_rate, subtype="FLOAT")
    stem.with_suffix(".json").write_text(json.dumps(p.to_dict(), indent=1))
    return str(stem.with_suffix(".wav"))


def generate(out_dir: str | Path, count: int, start_seed: int = 0, duration_s: float = 8.0,
             sample_rate: int = 44100, workers: int = 1) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(start_seed + i, duration_s, sample_rate, str(out_dir)) for i in range(count)]
    if workers <= 1:
        return [_render_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_render_one, jobs, chunksize=4))
