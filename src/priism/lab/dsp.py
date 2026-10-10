"""Effects shared by the synthetic sources (acid, skank, ...)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


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
