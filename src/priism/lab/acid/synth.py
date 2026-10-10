"""A small TB-303 style monosynth.

Saw or square oscillator into a resonant 4-pole low-pass ladder, with a
decaying filter envelope, accents and slides. It is not a circuit-accurate
emulation: the goal is lines a separation model cannot tell apart from the
acid lines in real records, with a wide spread of settings.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .params import AcidParams

try:  # Optional speed-up; the pure Python loop gives the same output.
    from numba import njit
except ImportError:  # pragma: no cover - depends on the environment
    njit = None


def midi_to_hz(note: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * 2.0 ** ((np.asarray(note) - 69) / 12.0)


def _poly_blep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Band-limiting correction around the discontinuity of a [0, 1) phase ramp."""
    out = np.zeros_like(t)
    a = t < dt
    x = t[a] / dt[a]
    out[a] = x + x - x * x - 1.0
    b = t > 1.0 - dt
    x = (t[b] - 1.0) / dt[b]
    out[b] = x * x + x + x + 1.0
    return out


def oscillator(freq_hz: np.ndarray, sample_rate: int, waveform: str) -> np.ndarray:
    dt = freq_hz / sample_rate
    phase = np.cumsum(dt) % 1.0
    saw = 2.0 * phase - 1.0 - _poly_blep(phase, dt)
    if waveform == "saw":
        return saw
    shifted = (phase + 0.5) % 1.0
    saw2 = 2.0 * shifted - 1.0 - _poly_blep(shifted, dt)
    return 0.5 * (saw - saw2)


def _ladder_py(x: np.ndarray, g: np.ndarray, k: float) -> np.ndarray:
    y = np.empty_like(x)
    y1 = y2 = y3 = y4 = 0.0
    tanh = math.tanh
    for n in range(x.shape[0]):
        gn = g[n]
        u = tanh(x[n] - k * y4)
        y1 += gn * (u - y1)
        y2 += gn * (y1 - y2)
        y3 += gn * (y2 - y3)
        y4 += gn * (y3 - y4)
        y[n] = y4
    return y


_ladder = njit(cache=True)(_ladder_py) if njit else _ladder_py


def ladder_filter(x: np.ndarray, cutoff_hz: np.ndarray, resonance: float, sample_rate: int) -> np.ndarray:
    """4-pole low-pass with a saturating input and resonance feedback.

    ``resonance`` in [0, 1] maps to a loop gain k in [0, 4); the input is
    scaled up with k to make up for the passband loss of a resonant ladder.
    """
    fc = np.clip(cutoff_hz, 20.0, 0.42 * sample_rate)
    g = 1.0 - np.exp(-2.0 * np.pi * fc / sample_rate)
    k = 3.95 * float(np.clip(resonance, 0.0, 1.0))
    return _ladder(np.ascontiguousarray(x * (1.0 + 0.5 * k), dtype=np.float64), g.astype(np.float64), k)


def _one_pole_smooth(x: np.ndarray, time_s: float, sample_rate: int) -> np.ndarray:
    a = math.exp(-1.0 / max(time_s * sample_rate, 1.0))
    return lfilter([1.0 - a], [1.0, -a], x)


def control_signals(p: AcidParams, n_samples: int) -> dict[str, np.ndarray]:
    """Per-sample pitch, gate, envelopes and level from the step pattern."""
    sr = p.sample_rate
    s = p.synth
    step_len = 60.0 / p.bpm / 4.0 * sr
    n_steps = int(math.ceil(n_samples / step_len))

    note = np.full(n_samples, float(p.steps[0].note))
    gate = np.zeros(n_samples)
    level = np.ones(n_samples)
    filt_env = np.zeros(n_samples)
    acc_env = np.zeros(n_samples)
    t_env = 0.0  # seconds since the last trigger
    env_peak = 0.0
    acc_peak = 0.0
    glide_len = max(1, int(s.glide_s * sr))
    prev_note = float(p.steps[0].note)

    for i in range(n_steps):
        step = p.steps[i % len(p.steps)]
        nxt = p.steps[(i + 1) % len(p.steps)]
        prev = p.steps[(i - 1) % len(p.steps)] if i > 0 else None
        a = int(round(i * step_len))
        b = min(int(round((i + 1) * step_len)), n_samples)
        if a >= b:
            break
        seg = b - a
        tied_in = prev is not None and prev.gate and prev.slide and step.gate

        if step.gate:
            target = float(step.note)
            if tied_in:
                m = min(glide_len, seg)
                note[a : a + m] = np.linspace(prev_note, target, m, endpoint=False)
                note[a + m : b] = target
            else:
                note[a:b] = target
                t_env = 0.0
                env_peak = 1.0
                acc_peak = s.accent_amount if step.accent else 0.0
            prev_note = target
            holds = step.slide and nxt.gate
            g_end = b if holds else a + max(1, int(seg * s.gate_fraction))
            gate[a:g_end] = 1.0
            level[a:b] = 1.0 + 0.8 * s.accent_amount * step.accent
        else:
            note[a:b] = prev_note
            level[a:b] = level[a - 1] if a > 0 else 1.0

        t = t_env + np.arange(seg) / sr
        filt_env[a:b] = env_peak * np.exp(-t / s.decay_s)
        acc_env[a:b] = acc_peak * np.exp(-t / 0.2)  # accents use a short, fixed decay
        t_env += seg / sr

    amp = _one_pole_smooth(gate * level, 0.004, sr)
    return {"note": note, "gate": gate, "amp": amp, "filt_env": filt_env, "acc_env": acc_env}


def render_dry(p: AcidParams) -> np.ndarray:
    """Mono dry synth output, before effects."""
    sr = p.sample_rate
    n = int(round(p.duration_s * sr))
    c = control_signals(p, n)
    freq = midi_to_hz(c["note"])
    osc = oscillator(freq, sr, p.synth.waveform)
    octaves = p.synth.env_mod_oct * c["filt_env"] + 2.0 * c["acc_env"]
    cutoff = p.synth.cutoff_hz * 2.0**octaves
    # The filter runs on the gated signal so resonance rings out naturally.
    y = ladder_filter(osc * c["amp"], cutoff, p.synth.resonance, sr)
    return y.astype(np.float32)
