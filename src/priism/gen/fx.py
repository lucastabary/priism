"""Per-source effects. Every effect (delay and reverb tails included) stays in its source's track."""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from ..acid.synth import ladder_filter
from ..dsp import feedback_delay, reverb_ir, saturate

CENTERED = {"kick", "sub", "bass", "snare", "clap"}
NO_SPACE = {"kick", "sub"}  # never sent to delay or reverb
# Studio effects from Spotify's pedalboard (compressor, chorus, phaser, bitcrush, Freeverb) on top of the
# effects below. Off by default: songs stay exactly as before. When on, the extra settings come from each source's
# own generator, so every other choice of a song (parts, levels, sends) stays the same; only the sound changes.
PB_FX = os.environ.get("PRIISM_PB_FX", "0") == "1"
CHORUS_KINDS = {"pad", "lead", "stab", "skank", "pluck", "arp", "acid", "bass"}
PHASER_KINDS = {"pad", "lead", "stab", "skank", "arp", "acid", "hat_closed", "hat_open", "ride", "noise_fx"}
DRUMLIKE = {"snare", "clap", "rim", "hat_closed", "hat_open", "ride", "crash", "tom", "cowbell", "clave", "conga",
            "shaker", "noise_fx"}


def sample_fx(kind: str, genre: str, rng: np.random.Generator) -> dict:
    u = rng.uniform
    dubby = genre in ("dub", "steppers", "dub_techno", "dubstep")
    tonal_hi = kind not in ("kick", "sub", "bass", "tom", "conga")
    delay_p = 0.0 if kind in NO_SPACE else (0.55 if dubby else 0.2) * (1.0 if tonal_hi else 0.4)
    reverb_p = 0.0 if kind in NO_SPACE else (0.6 if dubby else 0.35)
    # Real mixes keep most energy in the centre (side/mid energy ~0.05 on NI stems, see docs/ecart-synth-reel.md).
    pan = 0.0 if kind in CENTERED else float(np.clip(rng.normal(0, 0.25), -0.8, 0.8))
    return {
        "highpass_hz": float(np.exp(u(np.log(20), np.log(400)))) if kind not in ("kick", "sub", "bass", "tom") else 20.0,
        # Synths are often darker in real tracks than raw oscillators: a lowpass on most tonal sources.
        "lowpass_hz": float(np.exp(u(np.log(1500), np.log(9000)))) if (tonal_hi and kind not in DRUMLIKE
                                                                        and rng.random() < 0.6)
        else float(np.exp(u(np.log(5000), np.log(20000)))),
        "sweep_oct": float(u(1.5, 5.0)) if (tonal_hi and rng.random() < 0.2) else 0.0,  # automated filter opening
        "sweep_bars": float(rng.choice([4, 8, 16])),
        "drive": float(rng.choice([0.0, u(0.1, 1.0)], p=[0.7, 0.3])),
        "pan": pan,
        "width": float(u(0.0, 0.4)) if (kind not in CENTERED and rng.random() < 0.5) else 0.0,
        "delay_send": float(u(0.1, 0.6)) if rng.random() < delay_p else 0.0,
        "delay_steps_l": int(rng.choice([2, 3, 4, 6, 8])),
        "delay_steps_r": int(rng.choice([2, 3, 4, 6, 8])),
        "delay_feedback": float(u(0.2, 0.8 if dubby else 0.6)),
        "delay_lowpass_hz": float(u(1200, 7000)),
        "reverb_send": float(u(0.03, 0.3)) if rng.random() < reverb_p else 0.0,
        "reverb_rt60_s": float(u(0.4, 4.0)),
        "spring": bool(dubby and rng.random() < 0.5),
        "sidechain": float(u(0.3, 0.9)) if kind in ("bass", "sub", "pad", "stab", "acid", "arp", "lead", "skank") and rng.random() < 0.35 else 0.0,
    }


def sample_pb_fx(kind: str, genre: str, seed: int) -> dict:
    """Settings of the pedalboard effects for one source, from the source's own generator."""
    rng = np.random.default_rng([seed, 0x9B])
    u = rng.uniform
    dubby = genre in ("dub", "steppers", "dub_techno", "dubstep")
    d: dict = {}
    if kind != "noise_fx" and rng.random() < 0.5:  # most real parts are compressed
        d["comp"] = {"threshold_db": float(u(-30, -8)), "ratio": float(np.exp(u(np.log(1.5), np.log(10)))),
                     "attack_ms": float(np.exp(u(np.log(0.5), np.log(40)))),
                     "release_ms": float(np.exp(u(np.log(30), np.log(400))))}
    if kind in CHORUS_KINDS and rng.random() < (0.12 if kind == "bass" else 0.3):
        d["chorus"] = {"rate_hz": float(np.exp(u(np.log(0.1), np.log(3)))), "depth": float(u(0.1, 0.6)),
                       "centre_delay_ms": float(u(3, 15)), "feedback": float(u(0, 0.4)), "mix": float(u(0.2, 0.6))}
    if kind in PHASER_KINDS and rng.random() < (0.25 if dubby else 0.12):
        d["phaser"] = {"rate_hz": float(np.exp(u(np.log(0.05), np.log(2)))), "depth": float(u(0.4, 1.0)),
                       "centre_hz": float(np.exp(u(np.log(300), np.log(3000)))), "feedback": float(u(0, 0.7)),
                       "mix": float(u(0.3, 0.8))}
    if rng.random() < 0.05:  # lo-fi samplers and bitcrushers
        d["bitcrush_bits"] = float(u(5, 12))
    d["algo_reverb"] = bool(rng.random() < 0.5)  # when the reverb send is on: Freeverb instead of the noise IR
    d["algo_reverb_damping"] = float(u(0.2, 0.8))
    return d


def _pb(x: np.ndarray, plugins: list, sr: int) -> np.ndarray:
    """Run (n,) or (n, 2) float64 audio through pedalboard plugins; same shape and length out."""
    from pedalboard import Pedalboard

    mono = x.ndim == 1
    y = Pedalboard(plugins)(np.ascontiguousarray((x[None] if mono else x.T), np.float32), sr, reset=True)
    y = y.astype(np.float64)
    return y[0] if mono else y.T


def apply_fx(dry: np.ndarray, fx: dict, sr: int, bpm: float, seed: int) -> np.ndarray:
    """Mono dry track in, stereo (n, 2) track out. ``fx["pb"]`` (from ``sample_pb_fx``) adds pedalboard effects."""
    pb = fx.get("pb")
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
    if pb:
        from pedalboard import Bitcrush, Compressor

        mono = []
        if "comp" in pb:
            c = pb["comp"]
            mono.append(Compressor(c["threshold_db"], c["ratio"], c["attack_ms"], c["release_ms"]))
        if "bitcrush_bits" in pb:
            mono.append(Bitcrush(pb["bitcrush_bits"]))
        if mono:
            peak = np.max(np.abs(x)) + 1e-9  # the compressor threshold is relative to a full-scale part
            x = _pb(x / peak, mono, sr) * peak

    # Constant-power pan plus a small Haas offset for width.
    ang = (fx["pan"] + 1) * np.pi / 4
    left, right = np.cos(ang) * x, np.sin(ang) * x
    off = int(fx["width"] * 0.015 * sr)
    if off:
        right = np.concatenate([np.zeros(off), right[:-off]])
    out = np.stack([left, right], axis=1) * np.sqrt(2)
    if pb and ("chorus" in pb or "phaser" in pb):
        from pedalboard import Chorus, Phaser

        mods = []
        if "chorus" in pb:
            c = pb["chorus"]
            mods.append(Chorus(c["rate_hz"], c["depth"], c["centre_delay_ms"], c["feedback"], c["mix"]))
        if "phaser" in pb:
            c = pb["phaser"]
            mods.append(Phaser(c["rate_hz"], c["depth"], c["centre_hz"], c["feedback"], c["mix"]))
        out = _pb(out, mods, sr)

    step = 60.0 / bpm / 4.0 * sr
    if fx["delay_send"]:
        wet = np.stack([feedback_delay(x, int(fx["delay_steps_l"] * step), fx["delay_feedback"], fx["delay_lowpass_hz"], sr),
                        feedback_delay(x, int(fx["delay_steps_r"] * step), fx["delay_feedback"], fx["delay_lowpass_hz"], sr)],
                       axis=1)
        out = out + fx["delay_send"] * wet
    # A shared room (song.py, PRIISM_BUS_FX) sets the reverb for every source sent to it: same algorithm, same
    # IR. Reverbs are linear, so each source's share of the shared reverb stays in its own track.
    room = fx.get("room")
    algo = room["algo"] if room else bool(pb and pb["algo_reverb"])
    if fx["reverb_send"] and algo:
        from pedalboard import Reverb

        size = float(np.clip(fx["reverb_rt60_s"] / 4.0, 0.05, 0.98))
        damping = room["damping"] if room else pb["algo_reverb_damping"]
        wet = _pb(out, [Reverb(size, damping, wet_level=1.0, dry_level=0.0, width=0.5)], sr)
        out = out + fx["reverb_send"] * wet
    elif fx["reverb_send"]:
        ir = reverb_ir(fx["reverb_rt60_s"], sr, np.random.default_rng(room["seed"] if room else seed))
        if fx["spring"]:
            ir = sosfilt(butter(2, [300, 4500], btype="band", fs=sr, output="sos"), ir, axis=0)
            ir /= np.sqrt(np.sum(ir**2, axis=0, keepdims=True))
        wet = np.stack([fftconvolve(out[:, c], ir[:, c])[: len(out)] for c in range(2)], axis=1)
        out = out + fx["reverb_send"] * wet
    return out
