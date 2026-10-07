"""Effects that put a dry acid line in a realistic space.

The acid target stem keeps its effects: in a real record the delay and
reverb tails of the 303 belong to the 303, and a DJ muting the stem
expects them to go too.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .params import AcidParams


def saturate(x: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 0:
        return x
    gain = 1.0 + 4.0 * drive
    return np.tanh(gain * x) / np.tanh(gain)


def feedback_delay(x: np.ndarray, delay_samples: int, feedback: float, lowpass_hz: float, sample_rate: int) -> np.ndarray:
    """Wet signal of a feedback delay whose repeats get darker each pass."""
    sos = butter(2, lowpass_hz, btype="low", fs=sample_rate, output="sos")
    # y[n] = x[n - d] + fb * y[n - d], computed one delay-length block at a
    # time; repeats are darkened by low-passing the block fed back.
    d = max(1, delay_samples)
    y = np.zeros_like(x)
    zi = np.zeros((sos.shape[0], 2))
    for start in range(d, len(x), d):
        stop = min(start + d, len(x))
        src = x[start - d : stop - d] + feedback * y[start - d : stop - d]
        y[start:stop], zi = sosfilt(sos, src, zi=zi)
    return y


def reverb_ir(rt60_s: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Stereo impulse response: decaying noise with a darker tail."""
    n = int(rt60_s * sample_rate)
    t = np.arange(n) / sample_rate
    decay = np.exp(-6.9 * t / rt60_s)  # -60 dB at rt60
    ir = rng.standard_normal((n, 2)) * decay[:, None]
    sos = butter(1, 4000, btype="low", fs=sample_rate, output="sos")
    ir = sosfilt(sos, ir, axis=0)
    return ir / np.sqrt(np.sum(ir**2, axis=0, keepdims=True))


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
