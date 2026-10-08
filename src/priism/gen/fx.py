"""Per-source effects. Every effect (delay and reverb tails included) stays in its source's track."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from ..acid.synth import ladder_filter
from ..dsp import feedback_delay, reverb_ir, saturate

CENTERED = {"kick", "sub", "bass", "snare", "clap"}
NO_SPACE = {"kick", "sub"}  # never sent to delay or reverb


def sample_fx(kind: str, genre: str, rng: np.random.Generator) -> dict:
    u = rng.uniform
    dubby = genre in ("dub", "steppers", "dub_techno", "dubstep")
    tonal_hi = kind not in ("kick", "sub", "bass", "tom", "conga")
    delay_p = 0.0 if kind in NO_SPACE else (0.55 if dubby else 0.2) * (1.0 if tonal_hi else 0.4)
    reverb_p = 0.0 if kind in NO_SPACE else (0.6 if dubby else 0.35)
    pan = 0.0 if kind in CENTERED else float(np.clip(rng.normal(0, 0.45), -0.9, 0.9))
    return {
        "highpass_hz": float(np.exp(u(np.log(20), np.log(400)))) if kind not in ("kick", "sub", "bass", "tom") else 20.0,
        "lowpass_hz": float(np.exp(u(np.log(4000), np.log(20000)))),
        "sweep_oct": float(u(1.5, 5.0)) if (tonal_hi and rng.random() < 0.2) else 0.0,  # automated filter opening
        "sweep_bars": float(rng.choice([4, 8, 16])),
        "drive": float(rng.choice([0.0, u(0.1, 1.0)], p=[0.7, 0.3])),
        "pan": pan,
        "width": float(u(0.0, 0.7)) if kind not in CENTERED else 0.0,
        "delay_send": float(u(0.1, 0.6)) if rng.random() < delay_p else 0.0,
        "delay_steps_l": int(rng.choice([2, 3, 4, 6, 8])),
        "delay_steps_r": int(rng.choice([2, 3, 4, 6, 8])),
        "delay_feedback": float(u(0.2, 0.8 if dubby else 0.6)),
        "delay_lowpass_hz": float(u(1200, 7000)),
        "reverb_send": float(u(0.05, 0.5)) if rng.random() < reverb_p else 0.0,
        "reverb_rt60_s": float(u(0.4, 4.0)),
        "spring": bool(dubby and rng.random() < 0.5),
        "sidechain": float(u(0.3, 0.9)) if kind in ("bass", "sub", "pad", "stab", "acid", "arp", "lead", "skank") and rng.random() < 0.35 else 0.0,
    }


def apply_fx(dry: np.ndarray, fx: dict, sr: int, bpm: float, seed: int) -> np.ndarray:
    """Mono dry track in, stereo (n, 2) track out."""
    x = dry.astype(np.float64)
    lo = min(fx["lowpass_hz"], 0.45 * sr)
    if fx["highpass_hz"] > 25:
        x = sosfilt(butter(2, fx["highpass_hz"], btype="high", fs=sr, output="sos"), x)
    if lo < 0.44 * sr:
        x = sosfilt(butter(2, lo, btype="low", fs=sr, output="sos"), x)
    if fx["sweep_oct"]:
        period = fx["sweep_bars"] * 4 * 60.0 / bpm
        t = np.arange(len(x)) / sr
        ramp = (t % period) / period  # opens over the period, then snaps shut: a typical build
        cutoff = 200.0 * 2.0 ** (fx["sweep_oct"] * ramp)
        x = ladder_filter(x, np.minimum(cutoff, 0.42 * sr), 0.3, sr) * 1.5
    if fx["drive"]:
        peak = np.max(np.abs(x)) + 1e-9
        x = saturate(x / peak, fx["drive"]) * peak

    # Constant-power pan plus a small Haas offset for width.
    ang = (fx["pan"] + 1) * np.pi / 4
    left, right = np.cos(ang) * x, np.sin(ang) * x
    off = int(fx["width"] * 0.015 * sr)
    if off:
        right = np.concatenate([np.zeros(off), right[:-off]])
    out = np.stack([left, right], axis=1) * np.sqrt(2)

    step = 60.0 / bpm / 4.0 * sr
    if fx["delay_send"]:
        wet = np.stack([feedback_delay(x, int(fx["delay_steps_l"] * step), fx["delay_feedback"], fx["delay_lowpass_hz"], sr),
                        feedback_delay(x, int(fx["delay_steps_r"] * step), fx["delay_feedback"], fx["delay_lowpass_hz"], sr)],
                       axis=1)
        out = out + fx["delay_send"] * wet
    if fx["reverb_send"]:
        ir = reverb_ir(fx["reverb_rt60_s"], sr, np.random.default_rng(seed))
        if fx["spring"]:
            ir = sosfilt(butter(2, [300, 4500], btype="band", fs=sr, output="sos"), ir, axis=0)
            ir /= np.sqrt(np.sum(ir**2, axis=0, keepdims=True))
        wet = np.stack([fftconvolve(out[:, c], ir[:, c])[: len(out)] for c in range(2)], axis=1)
        out = out + fx["reverb_send"] * wet
    return out
