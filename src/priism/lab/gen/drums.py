"""Drum machine elements and their patterns, one source per element.

Voices are procedural models in the spirit of the 808/909/707 (oscillators,
noise, envelopes), with wide random settings so the separator does not learn
one particular machine. Each element is a separate track: a kick and a hat
from the same kit are two sources.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from ..dsp import saturate

# 808 cymbal oscillator frequencies; scaled per kit for variety.
METAL_HZ = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0])


def _env(n: int, sr: int, decay_s: float, attack_s: float = 0.0005) -> np.ndarray:
    t = np.arange(n) / sr
    return np.minimum(1.0, t / max(attack_s, 1e-5)) * np.exp(-t / max(decay_s, 1e-4))


def _bp(x: np.ndarray, lo: float, hi: float, sr: int, order: int = 2) -> np.ndarray:
    hi = min(hi, 0.45 * sr)
    lo = min(lo, hi * 0.9)
    return sosfilt(butter(order, [lo, hi], btype="band", fs=sr, output="sos"), x)


def _hp(x: np.ndarray, f: float, sr: int, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, min(f, 0.45 * sr), btype="high", fs=sr, output="sos"), x)


def _sine_sweep(n: int, sr: int, f_end: float, ratio: float, tau_s: float) -> np.ndarray:
    t = np.arange(n) / sr
    f = f_end * (1.0 + (ratio - 1.0) * np.exp(-t / tau_s))
    return np.sin(2 * np.pi * np.cumsum(f) / sr)


def _metal(n: int, sr: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    ph = rng.uniform(0, 1, len(METAL_HZ))
    return sum(np.sign(np.sin(2 * np.pi * (f * scale * t + p))) for f, p in zip(METAL_HZ, ph)) / len(METAL_HZ)


LAYERS = {"kick": "kick", "snare": "clap", "clap": "snare"}  # producers stack these into one sound
LAYER_P = 0.25


def sample_voice(kind: str, rng: np.random.Generator, allow_layer: bool = True) -> dict:
    """Random settings for one drum element (a 'kit piece').

    Kicks, snares and claps are sometimes two sounds always triggered together (a layer). A layer
    is one source: it plays as one, and nothing in the signal can split it.
    """
    v = _sample_voice(kind, rng)
    if allow_layer and kind in LAYERS and rng.random() < LAYER_P:
        v["layer_kind"] = LAYERS[kind]
        v["layer"] = _sample_voice(LAYERS[kind], rng)
        v["layer_gain"] = float(rng.uniform(0.3, 0.9))
    return v


def _sample_voice(kind: str, rng: np.random.Generator) -> dict:
    u = rng.uniform
    if kind == "kick":
        long = rng.random() < 0.35  # 808-style boom vs 909-style punch
        return {"f_end": u(38, 62), "ratio": u(2.5, 7.0), "pitch_tau": u(0.008, 0.05),
                "decay": u(0.35, 1.2) if long else u(0.12, 0.35), "click": u(0.0, 0.6), "drive": u(0.0, 1.5)}
    if kind == "snare":
        return {"f1": u(160, 260), "f2": u(280, 420), "tone": u(0.2, 0.7), "tone_decay": u(0.05, 0.15),
                "noise_decay": u(0.08, 0.3), "noise_lo": u(800, 2500), "noise_hi": u(5000, 12000), "drive": u(0, 0.8)}
    if kind == "clap":
        return {"bursts": int(rng.integers(3, 6)), "gap": u(0.006, 0.014), "tail": u(0.08, 0.35),
                "lo": u(700, 1500), "hi": u(2000, 5000)}
    if kind == "rim":
        return {"f": u(1200, 2200), "decay": u(0.008, 0.03), "noise": u(0.1, 0.6)}
    if kind in ("hat_closed", "hat_open", "ride", "crash"):
        decay = {"hat_closed": u(0.025, 0.09), "hat_open": u(0.18, 0.6), "ride": u(0.6, 1.8),
                 "crash": u(1.2, 3.0)}[kind]
        return {"scale": u(0.8, 1.6), "decay": decay, "metal": u(0.2, 0.9),
                "hp": u(5000, 9000) if kind.startswith("hat") else u(3000, 6000),
                "ring_hz": u(3000, 6000) if kind == "ride" else 0.0}
    if kind == "tom":
        return {"f_end": u(80, 260), "ratio": u(1.3, 2.0), "pitch_tau": u(0.03, 0.1), "decay": u(0.15, 0.5),
                "noise": u(0.0, 0.2)}
    if kind == "cowbell":
        return {"f1": u(500, 600), "f2": u(780, 860), "decay": u(0.06, 0.3)}
    if kind == "clave":
        return {"f": u(2000, 3000), "decay": u(0.015, 0.05)}
    if kind == "conga":
        return {"f_end": u(180, 420), "ratio": u(1.05, 1.4), "decay": u(0.08, 0.25), "slap": u(0.0, 0.5)}
    if kind == "shaker":
        return {"attack": u(0.01, 0.04), "decay": u(0.03, 0.09), "hp": u(4000, 9000)}
    raise KeyError(kind)


def render_voice(kind: str, v: dict, sr: int, rng: np.random.Generator) -> np.ndarray:
    """One hit at full velocity, mono."""
    y = _render_voice(kind, v, sr, rng)
    if "layer" in v:
        z = v["layer_gain"] * _render_voice(v["layer_kind"], v["layer"], sr, rng)
        n = max(len(y), len(z))
        y = np.pad(y, (0, n - len(y))) + np.pad(z, (0, n - len(z)))
    return y


def _render_voice(kind: str, v: dict, sr: int, rng: np.random.Generator) -> np.ndarray:
    noise = lambda n: rng.uniform(-1, 1, n)  # noqa: E731
    if kind == "kick":
        n = int((v["decay"] * 5 + 0.02) * sr)
        y = _sine_sweep(n, sr, v["f_end"], v["ratio"], v["pitch_tau"]) * _env(n, sr, v["decay"])
        m = int(0.004 * sr)
        y[:m] += v["click"] * _hp(noise(m), 2000, sr) * np.linspace(1, 0, m)
        return saturate(y, v["drive"])
    if kind == "snare":
        n = int(max(v["noise_decay"], v["tone_decay"]) * 6 * sr)
        t = np.arange(n) / sr
        tone = (np.sin(2 * np.pi * v["f1"] * t) + 0.6 * np.sin(2 * np.pi * v["f2"] * t)) * _env(n, sr, v["tone_decay"])
        nz = _bp(noise(n), v["noise_lo"], v["noise_hi"], sr) * _env(n, sr, v["noise_decay"])
        return saturate(v["tone"] * tone + (1 - v["tone"]) * 2.0 * nz, v["drive"])
    if kind == "clap":
        gap = int(v["gap"] * sr)
        n = gap * v["bursts"] + int(v["tail"] * 6 * sr)
        env = np.zeros(n)
        for b in range(v["bursts"]):
            seg = n - b * gap
            env[b * gap:] += (_env(seg, sr, v["tail"] if b == v["bursts"] - 1 else 0.006)) * (0.7 + 0.3 * rng.random())
        return _bp(noise(n), v["lo"], v["hi"], sr) * env * 2.0
    if kind == "rim":
        n = int(v["decay"] * 8 * sr)
        t = np.arange(n) / sr
        return (np.sin(2 * np.pi * v["f"] * t) + v["noise"] * _bp(noise(n), 1500, 6000, sr)) * _env(n, sr, v["decay"])
    if kind in ("hat_closed", "hat_open", "ride", "crash"):
        n = int(v["decay"] * 5 * sr)
        x = v["metal"] * _metal(n, sr, v["scale"], rng) + (1 - v["metal"]) * noise(n)
        if v["ring_hz"]:
            x = x + 0.3 * np.sin(2 * np.pi * v["ring_hz"] * np.arange(n) / sr)
        return _hp(x, v["hp"], sr, 4) * _env(n, sr, v["decay"]) * 1.5
    if kind == "tom":
        n = int(v["decay"] * 5 * sr)
        y = _sine_sweep(n, sr, v["f_end"], v["ratio"], v["pitch_tau"]) * _env(n, sr, v["decay"])
        return y + v["noise"] * _bp(noise(n), 500, 4000, sr) * _env(n, sr, 0.02)
    if kind == "cowbell":
        n = int(v["decay"] * 6 * sr)
        t = np.arange(n) / sr
        x = np.sign(np.sin(2 * np.pi * v["f1"] * t)) + np.sign(np.sin(2 * np.pi * v["f2"] * t))
        return _bp(x, 400, 3000, sr) * (0.6 * _env(n, sr, 0.015) + 0.4 * _env(n, sr, v["decay"]))
    if kind == "clave":
        n = int(v["decay"] * 6 * sr)
        return np.sin(2 * np.pi * v["f"] * np.arange(n) / sr) * _env(n, sr, v["decay"])
    if kind == "conga":
        n = int(v["decay"] * 6 * sr)
        y = _sine_sweep(n, sr, v["f_end"], v["ratio"], 0.02) * _env(n, sr, v["decay"])
        return y + v["slap"] * _bp(noise(n), 1000, 5000, sr) * _env(n, sr, 0.01)
    if kind == "shaker":
        n = int((v["attack"] + v["decay"] * 5) * sr)
        t = np.arange(n) / sr
        env = np.where(t < v["attack"], t / v["attack"], np.exp(-(t - v["attack"]) / v["decay"]))
        return _hp(noise(n), v["hp"], sr, 4) * env
    raise KeyError(kind)


# Base one-bar patterns per drum style: kind -> list of candidate step lists (16 steps per bar).
EIGHTHS = list(range(0, 16, 2))
OFFBEATS = [2, 6, 10, 14]
SIXTEENTHS = list(range(16))
FOUR = [0, 4, 8, 12]
STYLES: dict[str, dict[str, list[list[int]]]] = {
    "four_floor": {"kick": [FOUR], "clap": [[4, 12]], "snare": [[4, 12], [12]], "hat_closed": [OFFBEATS, SIXTEENTHS, EIGHTHS],
                   "hat_open": [OFFBEATS], "ride": [EIGHTHS, OFFBEATS]},
    "house": {"kick": [FOUR], "clap": [[4, 12]], "snare": [[4, 12]], "hat_closed": [SIXTEENTHS, EIGHTHS],
              "hat_open": [OFFBEATS], "ride": [OFFBEATS]},
    "electro": {"kick": [[0, 6, 10], [0, 3, 10], [0, 7, 10, 13], [0, 10]], "snare": [[4, 12]], "clap": [[4, 12]],
                "hat_closed": [EIGHTHS, SIXTEENTHS], "hat_open": [[6, 14], OFFBEATS], "ride": [EIGHTHS]},
    "breakbeat": {"kick": [[0, 10], [0, 3, 10], [0, 6, 10]], "snare": [[4, 12], [4, 7, 12, 15]], "clap": [[4, 12]],
                  "hat_closed": [EIGHTHS, SIXTEENTHS], "hat_open": [[6, 14], [14]], "ride": [EIGHTHS]},
    "dnb": {"kick": [[0, 10], [0, 10, 11], [0, 6, 10]], "snare": [[4, 12], [4, 12, 15]], "clap": [[4, 12]],
            "hat_closed": [EIGHTHS, SIXTEENTHS, [2, 6, 7, 10, 14, 15]], "hat_open": [[6, 14], [14]], "ride": [EIGHTHS]},
    "one_drop": {"kick": [[8]], "rim": [[8], [8, 15]], "snare": [[8]], "clap": [[8]],
                 "hat_closed": [EIGHTHS, OFFBEATS, SIXTEENTHS], "hat_open": [[6, 14], [14]], "ride": [EIGHTHS]},
    "steppers": {"kick": [FOUR], "rim": [[8], [4, 12]], "snare": [[8], [4, 12]], "clap": [[8]],
                 "hat_closed": [EIGHTHS, OFFBEATS, SIXTEENTHS], "hat_open": [OFFBEATS, [14]], "ride": [EIGHTHS]},
    "halftime": {"kick": [[0], [0, 10], [0, 3, 11]], "snare": [[8]], "clap": [[8]], "rim": [[8], [6, 14]],
                 "hat_closed": [EIGHTHS, SIXTEENTHS, [0, 3, 6, 8, 11, 14]], "hat_open": [[6, 14], [14]], "ride": [EIGHTHS]},
}


def _random_steps(rng: np.random.Generator, density: float, n: int = 16) -> list[int]:
    """Syncopated percussion: favours off-grid 16ths, never empty."""
    w = np.array([0.6 if i % 4 == 0 else (1.0 if i % 2 else 0.8) for i in range(n)])
    steps = [i for i in range(n) if rng.random() < density * w[i]]
    return steps or [int(rng.choice([3, 6, 10, 11, 14]))]


def base_pattern(kind: str, style: str, rng: np.random.Generator) -> list[int]:
    """A 2-bar (32-step) base pattern for one element."""
    cands = STYLES[style].get(kind)
    if kind == "crash":
        return [0]
    if cands is not None:
        bar = list(cands[int(rng.integers(len(cands)))])
        two = bar + [s + 16 for s in bar]
        if kind == "kick" and style in ("dnb", "breakbeat", "electro", "halftime") and rng.random() < 0.5:
            alt = cands[int(rng.integers(len(cands)))]  # second bar of a two-bar break
            two = bar + [s + 16 for s in alt]
        return sorted(set(two))
    density = {"rim": 0.25, "tom": 0.12, "cowbell": 0.2, "clave": 0.25, "conga": 0.35, "shaker": 0.9,
               "snare": 0.15, "clap": 0.15, "hat_open": 0.2, "ride": 0.5}.get(kind, 0.2)
    if kind == "shaker":
        return SIXTEENTHS + [s + 16 for s in SIXTEENTHS] if rng.random() < 0.6 else EIGHTHS + [s + 16 for s in EIGHTHS]
    one = _random_steps(rng, density)
    two = _random_steps(rng, density) if rng.random() < 0.4 else one
    return sorted(set(one + [s + 16 for s in two]))


def hits_for_bars(kind: str, base: list[int], bars: list[int], rng: np.random.Generator, accent_every: int = 4,
                  ghost_p: float = 0.0, drop_p: float = 0.03) -> list[tuple[int, float]]:
    """(global 16th step, velocity) for the bars where the element plays."""
    out = []
    accent = rng.uniform(0.0, 0.35)
    for bar in bars:
        if kind == "crash":
            if bar % 8 == 0:
                out.append((bar * 16, float(rng.uniform(0.7, 1.0))))
            continue
        for s in base:
            if s // 16 != bar % 2:
                continue
            if rng.random() < drop_p and kind not in ("kick",):
                continue
            step = bar * 16 + s % 16
            vel = 1.0 - accent * (s % accent_every != 0) + rng.normal(0, 0.05)
            out.append((step, float(np.clip(vel, 0.2, 1.0))))
        if ghost_p and kind in ("snare", "hat_closed"):
            for s in range(16):
                if rng.random() < ghost_p:
                    out.append((bar * 16 + s, float(rng.uniform(0.15, 0.35))))
    return sorted(set(out))


def render_track(kind: str, voice: dict, hits: list[tuple[int, float]], n: int, sr: int, bpm: float, swing: float,
                 rng: np.random.Generator, humanize_s: float = 0.003) -> np.ndarray:
    """Mono dry track of one element. A few variants of the hit are rendered so repeats are not identical."""
    variants = [render_voice(kind, voice, sr, rng) for _ in range(3)]
    step_s = 60.0 / bpm / 4.0
    y = np.zeros(n)
    for step, vel in hits:
        t = (step + swing * (step % 2)) * step_s + rng.normal(0, humanize_s)
        a = int(round(max(0.0, t) * sr))
        if a >= n:
            continue
        hit = variants[int(rng.integers(len(variants)))]
        m = min(len(hit), n - a)
        y[a:a + m] += vel * hit[:m]
    return y
