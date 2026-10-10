"""Effects that put a dry acid line in a realistic space.

The acid target stem keeps its effects: in a real record the delay and
reverb tails of the 303 belong to the 303, and a DJ muting the stem
expects them to go too.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ..dsp import feedback_delay, reverb_ir, saturate
from .params import AcidParams


def apply_fx(dry: np.ndarray, p: AcidParams) -> np.ndarray:
    """Mono dry line in, stereo (n, 2) line with effects out."""
    sr = p.sample_rate
    fx = p.fx
    rng = np.random.default_rng(p.seed + 1)
    x = saturate(dry.astype(np.float64), fx.drive)

    # Small stereo spread: a short Haas-style offset on one side.
    offset = int(fx.width * 0.012 * sr)
    left = x
    right = np.concatenate([np.zeros(offset), x[: len(x) - offset]]) if offset else x
    out = np.stack([left, right], axis=1)

    if fx.delay_mix > 0:
        step = 60.0 / p.bpm / 4.0 * sr
        wet = np.stack(
            [
                feedback_delay(x, int(fx.delay_steps_l * step), fx.delay_feedback, fx.delay_lowpass_hz, sr),
                feedback_delay(x, int(fx.delay_steps_r * step), fx.delay_feedback, fx.delay_lowpass_hz, sr),
            ],
            axis=1,
        )
        out = out + fx.delay_mix * wet

    if fx.reverb_mix > 0:
        ir = reverb_ir(fx.reverb_rt60_s, sr, rng)
        wet = np.stack([fftconvolve(out[:, c], ir[:, c])[: len(out)] for c in range(2)], axis=1)
        out = out + fx.reverb_mix * wet

    return out.astype(np.float32)
