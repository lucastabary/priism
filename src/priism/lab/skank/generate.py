"""Render a skank part: dry chords, echo throws and spring reverb, stereo."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from ..dsp import feedback_delay, reverb_ir, saturate
from .params import SkankParams, sample_params  # noqa: F401  (part of the source interface)
from .synth import render_dry

params_from_dict = SkankParams.from_dict


def render(p: SkankParams) -> np.ndarray:
    """Stereo float32 part normalised to ``p.peak_dbfs``. Echo and reverb tails are part of the target."""
    sr = p.sample_rate
    fx = p.fx
    dry, send = render_dry(p)
    dry = saturate(dry / (np.max(np.abs(dry)) + 1e-9), fx.drive)
    send = send / (np.max(np.abs(dry)) + 1e-9)

    offset = int(fx.width * 0.012 * sr)
    right = np.concatenate([np.zeros(offset), dry[: len(dry) - offset]]) if offset else dry
    out = np.stack([dry, right], axis=1)

    if np.any(send):
        step = int(60.0 / p.bpm / 4.0 * sr * fx.echo_steps)
        # Slightly different left and right times spread the repeats across the field.
        wet = np.stack([feedback_delay(send, step, fx.echo_feedback, fx.echo_lowpass_hz, sr),
                        feedback_delay(send, int(step * (1 + 0.05 * fx.width)), fx.echo_feedback, fx.echo_lowpass_hz, sr)],
                       axis=1)
        out = out + wet

    if fx.spring_mix > 0:
        rng = np.random.default_rng(p.seed + 1)
        ir = reverb_ir(fx.spring_rt60_s, sr, rng)
        # A spring is band-limited and boxy: keep the wet signal in the mids.
        ir = sosfilt(butter(2, [300, 4500], btype="band", fs=sr, output="sos"), ir, axis=0)
        ir /= np.sqrt(np.sum(ir**2, axis=0, keepdims=True))
        wet = np.stack([fftconvolve(out[:, c], ir[:, c])[: len(out)] for c in range(2)], axis=1)
        out = out + fx.spring_mix * wet

    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out * (10 ** (p.peak_dbfs / 20) / peak)
    return out.astype(np.float32)
