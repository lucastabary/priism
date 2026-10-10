"""Render synthetic acid lines to disk, one FLAC and one JSON per line."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fx import apply_fx
from .params import AcidParams, sample_params  # noqa: F401  (part of the source interface)
from .synth import render_dry

params_from_dict = AcidParams.from_dict


def render(p: AcidParams) -> np.ndarray:
    """Stereo float32 acid line, normalised to ``p.peak_dbfs``."""
    out = apply_fx(render_dry(p), p)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out * (10 ** (p.peak_dbfs / 20) / peak)
    return out.astype(np.float32)


def generate(out_dir: str | Path, count: int, start_seed: int = 0, duration_s: float = 8.0,
             sample_rate: int = 44100, workers: int = 1) -> list[str]:
    from ..sources import generate as generate_source

    return generate_source("acid", out_dir, count, start_seed, duration_s, sample_rate, workers)
