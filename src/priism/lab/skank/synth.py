"""Chord voices for the skank: plucked guitar, drawbar organ, damped piano, filtered stab.

None of them aims at a faithful emulation. What matters for training is a
wide spread of timbres that share what makes a skank a skank: short chords
on the offbeat, thin (no bass), often drenched in spring reverb and echo.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from ..acid.synth import midi_to_hz, oscillator
from .params import Hit, SkankParams, ToneParams

RELEASE_S = 0.06


def _damp(n: int, hold: int, tau_s: float, sr: int) -> np.ndarray:
    """1 while the chord is held, then a fast exponential release (palm mute, key up)."""
    env = np.ones(n)
    t = np.arange(n - hold) / sr
    env[hold:] = np.exp(-t / tau_s)
    return env


def guitar(f: float, n: int, hold: int, t: ToneParams, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Karplus-Strong string: a noise burst circulating in a damped delay line."""
    period = max(2, int(round(sr / f - 0.5)))
    burst = rng.uniform(-1, 1, period)
    a = 1.0 - 0.85 * t.brightness  # softer picks give a duller burst
    burst = lfilter([1 - a], [1, -a], burst)
    x = np.zeros(n)
    x[: min(period, n)] = burst[: min(period, n)]
    den = np.zeros(period + 2)
    den[0] = 1.0
    den[period] = den[period + 1] = -t.damping / 2
    return lfilter([1.0], den, x) * _damp(n, hold, 0.012, sr)


ORGAN_HARMONICS = [1, 2, 3, 4, 6, 8]


def organ(f: float, n: int, hold: int, t: ToneParams, sr: int, rng: np.random.Generator) -> np.ndarray:
    tt = np.arange(n) / sr
    y = sum(level * np.sin(2 * np.pi * f * h * tt + rng.uniform(0, 2 * np.pi))
            for h, level in zip(ORGAN_HARMONICS, t.drawbars) if f * h < 0.45 * sr)
    y = y / max(sum(t.drawbars), 1e-3)
    attack = np.minimum(1.0, tt / 0.003)
    click = np.zeros(n)
    m = min(n, int(0.002 * sr))
    click[:m] = rng.uniform(-1, 1, m) * 0.3 * t.brightness  # key click
    return (y * attack + click) * _damp(n, hold, 0.015, sr)


def piano(f: float, n: int, hold: int, t: ToneParams, sr: int, rng: np.random.Generator) -> np.ndarray:
    tt = np.arange(n) / sr
    b = 1e-4  # string inharmonicity
    y = np.zeros(n)
    for k in range(1, 11):
        fk = k * f * np.sqrt(1 + b * k * k)
        if fk > 0.45 * sr:
            break
        y += k ** -(2.2 - 1.2 * t.brightness) * np.exp(-tt * (1 + 0.6 * k)) * np.sin(2 * np.pi * fk * tt)
    return y * np.minimum(1.0, tt / 0.001) * _damp(n, hold, 0.04, sr)


def stab(f: float, n: int, hold: int, t: ToneParams, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Two detuned saws through a low-pass: the dub techno chord."""
    spread = 2 ** (t.detune_cents / 1200 / 2)
    y = sum(oscillator(np.full(n, f * s), sr, "saw") for s in (spread, 1 / spread)) / 2
    cutoff = 400 + 3500 * t.brightness
    y = sosfilt(butter(2, min(cutoff, 0.45 * sr), btype="low", fs=sr, output="sos"), y)
    tt = np.arange(n) / sr
    return y * np.minimum(1.0, tt / 0.002) * np.exp(-tt / 0.4) * _damp(n, hold, 0.03, sr)


VOICES = {"guitar": guitar, "organ": organ, "piano": piano, "stab": stab}


def render_hit(h: Hit, t: ToneParams, sr: int, rng: np.random.Generator) -> np.ndarray:
    hold = int(h.length_s * sr)
    n_note = hold + int(RELEASE_S * sr)
    strum = int(abs(h.strum_s) * sr)
    notes = h.notes if h.strum_s >= 0 else h.notes[::-1]  # downstroke starts on the low string
    out = np.zeros(n_note + strum * len(notes))
    for i, note in enumerate(notes):
        y = VOICES[t.instrument](float(midi_to_hz(note)), n_note, hold, t, sr, rng)
        out[i * strum : i * strum + n_note] += y
    return out * h.velocity / np.sqrt(len(notes))


def render_dry(p: SkankParams) -> tuple[np.ndarray, np.ndarray]:
    """Mono dry part and the echo send (throws send the whole hit)."""
    sr = p.sample_rate
    n = int(round(p.duration_s * sr))
    rng = np.random.default_rng(p.seed + 2)
    dry, send = np.zeros(n), np.zeros(n)
    for h in p.hits:
        y = render_hit(h, p.tone, sr, rng)
        a = int(round(h.time_s * sr))
        m = min(len(y), n - a)
        if m <= 0:
            continue
        dry[a : a + m] += y[:m]
        send[a : a + m] += y[:m] * (1.0 if h.throw else p.fx.echo_send)
    sos = butter(2, [p.tone.highpass_hz, min(p.tone.lowpass_hz, 0.45 * sr)], btype="band", fs=sr, output="sos")
    return sosfilt(sos, dry), sosfilt(sos, send)
