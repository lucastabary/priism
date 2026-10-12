"""Deep, varied effect chains for the generator (Lucas, 2026-10-12): many effect types, composed in random
orders, repeated, automated within sections, plus descriptors measured on the rendered tracks.

``fx.apply_fx`` is one fixed chain (filters, drive, compressor, pan, chorus/phaser, one delay, one reverb):
a model can learn that signature. Here a source's chain is a list of nodes drawn per category (filters,
drives, dynamics, colour EQ, modulation, delays, reverbs, pan), mostly in the usual studio order (dynamics
and colour, then modulation, then space) and in a random order for ``EXPLORE_P`` of the chains; categories
may repeat (two drives, filter -> drive -> filter, two delays). Continuous parameters may follow a curve
within a section (LFO, ramp, steps): that is the per-frame V(t). Chains that change between sections give
the per-section V̄, measured on the rendered track (``describe``) since composed chains make knob values
ill-posed targets.

Every effect stays in its source's track (the mix is still the sum of the tracks). Strict-loop songs never
use these chains: automation would break the loops.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

from ..acid.synth import ladder_filter
from ..dsp import feedback_delay

EXPLORE_P = 0.2  # share of chains in a fully random order (reverb before drive, delay into a filter...)
AUTO_P = 0.5  # share of chains with automated parameters
NO_SPACE = {"kick", "sub"}
CENTERED = {"kick", "sub", "bass", "snare", "clap"}
# Real impulse responses (e.g. the Voxengo IM Reverbs pack): a folder of WAV files, used by "ir" reverbs.
IR_DIR = os.environ.get("PRIISM_IR_DIR", "")
_IRS: dict = {}

# Category -> (rank in the usual order, node types). Filters appear twice: before the drive (tone shaping)
# and after it (colour), so the usual order already gives filter -> drive -> filter.
CATEGORIES = {
    "filter": (0, ["hp", "lp", "ladder", "svf", "comb"]),
    "drive": (1, ["tube", "foldback", "hardclip", "bitcrush", "downsample"]),
    "dynamics": (2, ["comp"]),
    "colour": (3, ["eq_bell", "shelf", "lp", "hp", "svf"]),
    "pan": (4, ["pan"]),
    "modulation": (5, ["chorus", "phaser", "flanger", "tremolo", "autopan", "ringmod"]),
    "delay": (6, ["stereo_delay", "pingpong", "tape", "multitap", "slapback"]),
    "reverb": (7, ["room", "hall", "plate", "gated", "spring", "freeverb", "shimmer", "ir"]),
}
SPACE = {"stereo_delay", "pingpong", "tape", "multitap", "slapback", "room", "hall", "plate", "gated", "spring",
         "freeverb", "shimmer", "ir"}
# Automatable parameters: "log" ones move in octaves (depth = octaves), "lin" ones by +- depth.
AUTOMATABLE = {"hp": {"cutoff_hz": "log"}, "lp": {"cutoff_hz": "log"}, "ladder": {"cutoff_hz": "log"},
               "tube": {"drive": "lin"}, "foldback": {"drive": "lin"}, "hardclip": {"drive": "lin"},
               "pan": {"pan": "lin"}, **{t: {"send": "lin"} for t in SPACE}}


# --------------------------------------------------------------------------------------------- sampling

def _node(t: str, kind: str, rng: np.random.Generator) -> dict:
    """One effect node with random settings."""
    u = rng.uniform
    lg = lambda a, b: float(np.exp(u(np.log(a), np.log(b))))  # noqa: E731
    p: dict = {}
    if t == "hp":
        p = {"cutoff_hz": lg(25, 800), "order": int(rng.choice([1, 2, 4]))}
    elif t == "lp":
        p = {"cutoff_hz": lg(700, 16000), "order": int(rng.choice([1, 2, 4]))}
    elif t == "ladder":
        p = {"cutoff_hz": lg(300, 9000), "resonance": float(u(0.0, 0.8))}
    elif t == "svf":
        p = {"mode": str(rng.choice(["LPF12", "HPF12", "BPF12", "LPF24", "BPF24"])), "cutoff_hz": lg(200, 8000),
             "resonance": float(u(0.0, 0.7))}
    elif t == "comb":
        p = {"delay_ms": lg(0.5, 15), "gain": float(u(-0.8, 0.8)), "feedback": bool(rng.random() < 0.4)}
    elif t == "tube":
        p = {"drive": float(u(0.1, 1.0)), "bias": float(u(0.0, 0.4)), "mix": float(u(0.5, 1.0))}
    elif t == "foldback":
        p = {"drive": float(u(0.1, 1.0)), "mix": float(u(0.3, 1.0))}
    elif t == "hardclip":
        p = {"drive": float(u(0.1, 1.0)), "mix": float(u(0.5, 1.0))}
    elif t == "bitcrush":
        p = {"bits": float(u(4, 12)), "mix": float(u(0.3, 1.0))}
    elif t == "downsample":
        p = {"factor": int(rng.choice([2, 3, 4, 6, 8, 12, 16])), "mix": float(u(0.3, 1.0))}
    elif t == "comp":
        p = {"threshold_db": float(u(-30, -6)), "ratio": lg(1.5, 12), "attack_ms": lg(0.3, 40),
             "release_ms": lg(30, 500)}
    elif t == "eq_bell":
        p = {"cutoff_hz": lg(80, 10000), "gain_db": float(u(-12, 12)), "q": lg(0.3, 6)}
    elif t == "shelf":
        p = {"high": bool(rng.random() < 0.5), "cutoff_hz": lg(100, 8000), "gain_db": float(u(-12, 9))}
    elif t == "pan":
        p = {"pan": 0.0 if kind in CENTERED else float(np.clip(rng.normal(0, 0.35), -0.9, 0.9)),
             "haas_ms": float(u(0, 15)) if kind not in CENTERED and rng.random() < 0.4 else 0.0}
    elif t == "chorus":
        p = {"rate_hz": lg(0.1, 3), "depth": float(u(0.1, 0.7)), "centre_delay_ms": float(u(5, 20)),
             "feedback": float(u(0, 0.4)), "mix": float(u(0.2, 0.6))}
    elif t == "flanger":
        p = {"rate_hz": lg(0.05, 1.5), "depth": float(u(0.3, 1.0)), "centre_delay_ms": float(u(0.8, 3)),
             "feedback": float(u(0.3, 0.8)), "mix": float(u(0.3, 0.6))}
    elif t == "phaser":
        p = {"rate_hz": lg(0.05, 2), "depth": float(u(0.4, 1.0)), "centre_hz": lg(300, 3000),
             "feedback": float(u(0, 0.7)), "mix": float(u(0.3, 0.8))}
    elif t == "tremolo":
        p = {"steps": float(rng.choice([1, 2, 4, 8, 16])), "depth": float(u(0.2, 0.9)),
             "shape": str(rng.choice(["sine", "square"]))}
    elif t == "autopan":
        p = {"steps": float(rng.choice([2, 4, 8, 16, 32])), "depth": float(u(0.2, 0.9))}
    elif t == "ringmod":
        p = {"freq_hz": lg(30, 2000), "mix": float(u(0.1, 0.6))}
    elif t == "stereo_delay":
        p = {"steps_l": int(rng.choice([2, 3, 4, 6, 8])), "steps_r": int(rng.choice([2, 3, 4, 6, 8])),
             "feedback": float(u(0.2, 0.7)), "lowpass_hz": lg(1200, 8000), "send": float(u(0.1, 0.6))}
    elif t == "pingpong":
        p = {"steps": int(rng.choice([2, 3, 4, 6])), "feedback": float(u(0.3, 0.75)), "lowpass_hz": lg(1500, 8000),
             "send": float(u(0.1, 0.6))}
    elif t == "tape":
        p = {"steps": int(rng.choice([3, 4, 6, 8])), "feedback": float(u(0.3, 0.85)), "lowpass_hz": lg(1000, 5000),
             "wow_hz": lg(0.2, 2.0), "wow_ms": float(u(0.5, 4.0)), "drive": float(u(0.1, 0.8)),
             "send": float(u(0.1, 0.6))}
    elif t == "multitap":
        p = {"taps": [[int(rng.integers(1, 9)), float(u(-1, 1)), float(u(0.2, 0.8))]
                      for _ in range(int(rng.integers(2, 5)))], "send": float(u(0.1, 0.5))}
    elif t == "slapback":
        p = {"delay_ms": float(u(60, 160)), "send": float(u(0.1, 0.5))}
    elif t in ("room", "hall", "plate", "gated", "spring"):
        rt = {"room": (0.2, 1.0), "hall": (1.5, 5.0), "plate": (0.8, 3.0), "gated": (1.0, 3.0), "spring": (1.0, 3.0)}[t]
        p = {"rt60_s": float(u(*rt)), "predelay_ms": float(u(0, 60 if t == "hall" else 15)),
             "damping_hz": lg(2500, 12000), "seed": int(rng.integers(2**31)), "send": float(u(0.05, 0.4))}
        if t == "gated":
            p["gate_ms"] = float(u(120, 450))
    elif t == "freeverb":
        p = {"size": float(u(0.2, 0.95)), "damping": float(u(0.2, 0.8)), "width": float(u(0.3, 1.0)),
             "send": float(u(0.05, 0.4))}
    elif t == "shimmer":
        p = {"rt60_s": float(u(2.0, 6.0)), "seed": int(rng.integers(2**31)), "amount": float(u(0.2, 0.6)),
             "send": float(u(0.05, 0.3))}
    elif t == "ir":
        p = {"index": int(rng.integers(1 << 30)), "send": float(u(0.05, 0.4))}
    if kind in NO_SPACE and "send" in p:
        p["send"] = float(p["send"] * 0.3)
    return {"type": t, "params": p}


def _irs() -> list[Path]:
    if "list" not in _IRS:
        _IRS["list"] = sorted(Path(IR_DIR).glob("*.wav")) if IR_DIR else []
    return _IRS["list"]


def sample_chain(kind: str, rng: np.random.Generator, n_bars: int) -> list[dict]:
    """A random chain for one source: node counts per category (repeats allowed), the usual order or, for
    EXPLORE_P of the chains, a random one; sometimes automated parameters (``_automate``)."""
    probs = {"filter": (0.55, 0.15), "drive": (0.4, 0.15), "dynamics": (0.4, 0.0), "colour": (0.5, 0.2),
             "pan": (1.0, 0.0), "modulation": (0.3, 0.08), "delay": (0.3, 0.08), "reverb": (0.5, 0.1)}
    if kind in NO_SPACE:
        probs.update(delay=(0.05, 0.0), reverb=(0.15, 0.0), modulation=(0.1, 0.0))
    nodes = []
    for cat, (p1, p2) in probs.items():
        k = (rng.random() < p1) + (rng.random() < p2)
        types = [t for t in CATEGORIES[cat][1] if t != "ir" or _irs()]
        for _ in range(int(k)):
            n = _node(str(rng.choice(types)), kind, rng)
            n["category"] = cat
            nodes.append(n)
    if rng.random() < EXPLORE_P:
        nodes = [nodes[i] for i in rng.permutation(len(nodes))]
        order = "explore"
    else:
        nodes.sort(key=lambda n: CATEGORIES[n["category"]][0])
        order = "usual"
    if rng.random() < AUTO_P:
        _automate(nodes, rng, n_bars, kind)
    return [{"order": order}] + nodes


def _automate(nodes: list[dict], rng: np.random.Generator, n_bars: int, kind: str = "") -> None:
    """Give 1 or 2 automatable nodes a curve on one parameter (edits the nodes in place). Centred kinds
    (kick, bass...) never get an automated pan."""
    cands = [n for n in nodes if n["type"] in AUTOMATABLE and not (n["type"] == "pan" and kind in CENTERED)]
    for n in [cands[i] for i in rng.permutation(len(cands))][: int(rng.integers(1, 3))]:
        param, scale = next(iter(AUTOMATABLE[n["type"]].items()))
        shape = str(rng.choice(["lfo", "ramp", "steps"]))
        a = {"param": param, "shape": shape, "scale": scale,
             "depth": float(rng.uniform(0.5, 3.0) if scale == "log" else rng.uniform(0.2, 0.8)),
             "period_bars": float(rng.choice([1, 2, 4, 8, 16])), "phase": float(rng.uniform(0, 1)),
             "wave": str(rng.choice(["sine", "triangle"])), "up": bool(rng.random() < 0.7),
             "seed": int(rng.integers(2**31))}
        n["auto"] = a


def curve(a: dict, t: np.ndarray, bpm: float) -> np.ndarray:
    """Curve in [-1, 1] of one automation at times ``t`` (s). "lfo": a sine or triangle over ``period_bars``;
    "ramp": a rise (or fall) over each ``period_bars``, as a build; "steps": a new random value every bar."""
    bar_s = 240.0 / bpm
    ph = (t / (a["period_bars"] * bar_s) + a["phase"]) % 1.0
    if a["shape"] == "lfo":
        return np.sin(2 * np.pi * ph) if a["wave"] == "sine" else 1 - 4 * np.abs(ph - 0.5)
    if a["shape"] == "ramp":
        r = 2 * ph - 1
        return r if a["up"] else -r
    vals = np.random.default_rng(a["seed"]).uniform(-1, 1, int(t.max() / bar_s) + 2)
    return vals[(t / bar_s).astype(int)]


def _param(n: dict, name: str, t: np.ndarray | None, bpm: float, lo: float, hi: float):
    """A node parameter: a scalar, or an array over ``t`` when it is automated."""
    base = n["params"][name]
    a = n.get("auto")
    if a is None or a["param"] != name or t is None:
        return base
    c = curve(a, t, bpm)
    v = base * 2.0 ** (a["depth"] * c) if a["scale"] == "log" else base + a["depth"] * c
    return np.clip(v, lo, hi)


# --------------------------------------------------------------------------------------------- rendering

def _pb(x: np.ndarray, plugins: list, sr: int) -> np.ndarray:
    from pedalboard import Pedalboard

    return Pedalboard(plugins)(np.ascontiguousarray(x.T, np.float32), sr, reset=True).T.astype(np.float64)


_SOS: dict = {}


def _butter(order: int, fc: float, btype: str, sr: int) -> np.ndarray:
    """Butterworth sections, cached on a 1/24-octave grid of cutoffs (automated filters redesign per block)."""
    key = (order, round(24 * np.log2(fc)), btype, sr)
    if key not in _SOS:
        _SOS[key] = butter(order, 2 ** (key[1] / 24), btype=btype, fs=sr, output="sos")
    return _SOS[key]


def _filter_tv(x: np.ndarray, btype: str, fc, order: int, sr: int, block: int = 256) -> np.ndarray:
    """Butterworth filter whose cutoff may change over time (one design per block, state carried over)."""
    if np.isscalar(fc):
        return sosfilt(butter(order, min(fc, 0.45 * sr), btype=btype, fs=sr, output="sos"), x, axis=0)
    out = np.empty_like(x)
    zi = None
    for a in range(0, len(x), block):
        sos = _butter(order, float(np.clip(fc[a], 20, 0.45 * sr)), btype, sr)
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, x.shape[1]))
        out[a:a + block], zi = sosfilt(sos, x[a:a + block], axis=0, zi=zi)
    return out


def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) + 1e-9


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    return np.concatenate([np.zeros((d,) + x.shape[1:]), x[: len(x) - d]]) if 0 < d < len(x) else (
        x if d <= 0 else np.zeros_like(x))


def _noise_ir(p: dict, t: str, sr: int) -> np.ndarray:
    """Stereo IR of the algorithmic reverbs: decaying noise, plus early reflections for rooms and halls, a
    dense bright build for plates, a hard gate for gated reverbs, a band-passed sound for springs."""
    rng = np.random.default_rng(p["seed"])
    rt = p["rt60_s"]
    n = int(min(rt, 6.0) * sr)
    tt = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * tt / rt)[:, None]
    if t == "gated":
        g = p["gate_ms"] / 1000
        ir *= np.clip((g - tt) / 0.02, 0, 1)[:, None]
    if t in ("room", "hall"):
        for _ in range(int(rng.integers(4, 12))):
            k = int(rng.uniform(0.002, 0.08 if t == "hall" else 0.03) * sr)
            if k < n:
                ir[k] += rng.uniform(0.5, 2.0, 2) * rng.choice([-1, 1], 2) * 8
    ir = sosfilt(butter(1, min(p["damping_hz"], 0.45 * sr), btype="low", fs=sr, output="sos"), ir, axis=0)
    if t == "spring":
        ir = sosfilt(butter(2, [300, 4500], btype="band", fs=sr, output="sos"), ir, axis=0)
    ir = np.concatenate([np.zeros((int(p.get("predelay_ms", 0) / 1000 * sr), 2)), ir])
    return ir / np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True) + 1e-12)


def _file_ir(p: dict, sr: int) -> np.ndarray:
    import soundfile as sf

    files = _irs()
    path = files[p["index"] % len(files)]
    key = (str(path), sr)
    if key not in _IRS:
        ir, fsr = sf.read(path, dtype="float64", always_2d=True)
        ir = ir[:, :2] if ir.shape[1] >= 2 else np.repeat(ir, 2, axis=1)
        if fsr != sr:
            g = np.gcd(int(fsr), int(sr))
            ir = resample_poly(ir, sr // g, fsr // g, axis=0)
        _IRS[key] = ir / np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True) + 1e-12)
    return _IRS[key]


def _conv(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    return np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)


def _apply_node(x: np.ndarray, n: dict, sr: int, bpm: float, t: np.ndarray | None, space: bool = True) -> np.ndarray:
    """One node on a stereo signal (n, 2). ``space`` False mutes delays and reverbs (the direct path)."""
    ty, p = n["type"], n["params"]
    step = 60.0 / bpm / 4.0 * sr
    if ty in SPACE and not space:
        return x
    if ty in ("hp", "lp"):
        return _filter_tv(x, "high" if ty == "hp" else "low", _param(n, "cutoff_hz", t, bpm, 20, 0.45 * sr),
                          p["order"], sr)
    if ty == "ladder":
        fc = _param(n, "cutoff_hz", t, bpm, 20, 0.42 * sr)
        fc = np.full(len(x), fc) if np.isscalar(fc) else fc
        return np.stack([ladder_filter(x[:, c], fc, p["resonance"], sr) for c in range(2)], axis=1)
    if ty == "svf":
        from pedalboard import LadderFilter

        mode = getattr(LadderFilter.Mode, p["mode"])
        return _pb(x, [LadderFilter(mode, p["cutoff_hz"], p["resonance"])], sr)
    if ty == "comb":
        d = max(1, int(p["delay_ms"] / 1000 * sr))
        if p["feedback"]:
            a = np.zeros(d + 1)
            a[0], a[d] = 1.0, -p["gain"] * 0.9
            return lfilter([1.0], a, x, axis=0) * (1 - abs(p["gain"]) * 0.9)
        return (x + p["gain"] * _shift(x, d)) / (1 + abs(p["gain"]))
    if ty in ("tube", "foldback", "hardclip"):
        pk = _peak(x)
        g = 1.0 + 6.0 * np.asarray(_param(n, "drive", t, bpm, 0.0, 1.0))
        g = g[:, None] if np.ndim(g) else g
        z = x / pk * g
        if ty == "tube":
            b = p["bias"]
            y = np.tanh(z + b) - np.tanh(b)
        elif ty == "foldback":
            y = np.abs(((z - 1) % 4) - 2) - 1
        else:
            y = np.clip(z, -1, 1)
        y = y / (_peak(y)) * pk
        return (1 - p["mix"]) * x + p["mix"] * y
    if ty == "bitcrush":
        from pedalboard import Bitcrush

        pk = _peak(x)
        return (1 - p["mix"]) * x + p["mix"] * _pb(x / pk, [Bitcrush(p["bits"])], sr) * pk
    if ty == "downsample":
        f = p["factor"]
        held = np.repeat(x[::f], f, axis=0)[: len(x)]
        return (1 - p["mix"]) * x + p["mix"] * held
    if ty == "comp":
        from pedalboard import Compressor

        pk = _peak(x)
        return _pb(x / pk, [Compressor(p["threshold_db"], p["ratio"], p["attack_ms"], p["release_ms"])], sr) * pk
    if ty == "eq_bell":
        from pedalboard import PeakFilter

        return _pb(x, [PeakFilter(min(p["cutoff_hz"], 0.45 * sr), p["gain_db"], p["q"])], sr)
    if ty == "shelf":
        from pedalboard import HighShelfFilter, LowShelfFilter

        cls = HighShelfFilter if p["high"] else LowShelfFilter
        return _pb(x, [cls(min(p["cutoff_hz"], 0.45 * sr), p["gain_db"])], sr)
    if ty == "pan":
        pan = _param(n, "pan", t, bpm, -1.0, 1.0)
        ang = (np.asarray(pan) + 1) * np.pi / 4
        m = x.mean(axis=1)
        left, right = np.cos(ang) * m, np.sin(ang) * m
        d = int(p["haas_ms"] / 1000 * sr)
        if d:
            right = _shift(right, d)
        return np.stack([left, right], axis=1) * np.sqrt(2)
    if ty in ("chorus", "flanger"):
        from pedalboard import Chorus

        return _pb(x, [Chorus(p["rate_hz"], p["depth"], p["centre_delay_ms"], p["feedback"], p["mix"])], sr)
    if ty == "phaser":
        from pedalboard import Phaser

        return _pb(x, [Phaser(p["rate_hz"], p["depth"], p["centre_hz"], p["feedback"], p["mix"])], sr)
    tt = np.arange(len(x)) / sr if t is None else t
    if ty == "tremolo":
        ph = (tt / (p["steps"] * step / sr)) % 1.0
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * ph) if p["shape"] == "sine" else (ph < 0.5).astype(float)
        return x * (1 - p["depth"] * lfo)[:, None]
    if ty == "autopan":
        ph = (tt / (p["steps"] * step / sr)) % 1.0
        ang = (p["depth"] * np.sin(2 * np.pi * ph) + 1) * np.pi / 4
        m = x.mean(axis=1)
        return np.stack([np.cos(ang) * m, np.sin(ang) * m], axis=1) * np.sqrt(2)
    if ty == "ringmod":
        return (1 - p["mix"]) * x + p["mix"] * x * np.sin(2 * np.pi * p["freq_hz"] * tt)[:, None]
    send = np.asarray(_param(n, "send", t, bpm, 0.0, 1.0))
    send = send[:, None] if send.ndim else send
    m = x.mean(axis=1)
    if ty == "stereo_delay":
        wet = np.stack([feedback_delay(m, int(p["steps_l"] * step), p["feedback"], p["lowpass_hz"], sr),
                        feedback_delay(m, int(p["steps_r"] * step), p["feedback"], p["lowpass_hz"], sr)], axis=1)
    elif ty == "pingpong":
        d = int(p["steps"] * step)
        both = feedback_delay(m, 2 * d, p["feedback"] ** 2, p["lowpass_hz"], sr)
        wet = np.stack([both, _shift(both, d) * p["feedback"] + _shift(m, d)], axis=1)
    elif ty == "tape":
        d = int(p["steps"] * step)
        w = feedback_delay(m, d, p["feedback"], p["lowpass_hz"], sr)
        # Wow: read the repeats with a slowly moving delay; then tape saturation.
        lag = p["wow_ms"] / 1000 * sr * (1 + np.sin(2 * np.pi * p["wow_hz"] * tt)) / 2
        idx = np.clip(np.arange(len(w)) - lag, 0, len(w) - 1)
        w = np.interp(idx, np.arange(len(w)), w)
        g = 1 + 3 * p["drive"]
        w = np.tanh(g * w / _peak(w)) / np.tanh(g) * _peak(w)
        wet = np.stack([w, w], axis=1)
    elif ty == "multitap":
        wet = np.zeros_like(x)
        for steps, pan, gain in p["taps"]:
            tap = _shift(m, int(steps * step)) * gain
            ang = (pan + 1) * np.pi / 4
            wet += np.stack([np.cos(ang) * tap, np.sin(ang) * tap], axis=1) * np.sqrt(2)
    elif ty == "slapback":
        tap = _shift(m, int(p["delay_ms"] / 1000 * sr))
        wet = np.stack([tap, tap], axis=1)
    elif ty in ("room", "hall", "plate", "gated", "spring"):
        wet = _conv(x, _noise_ir(p, ty, sr))
    elif ty == "freeverb":
        from pedalboard import Reverb

        wet = _pb(x, [Reverb(p["size"], p["damping"], wet_level=1.0, dry_level=0.0, width=p["width"])], sr)
    elif ty == "shimmer":
        # A long hall whose tail is fed back an octave up. Octave by full-wave rectification (the octave-fuzz
        # trick, high-passed to drop the DC): on a diffuse tail it sounds like the pitch-shifted feedback of a
        # shimmer, at a fraction of the cost of a real pitch shifter (~4 s per 26 s track).
        ir = _noise_ir({"rt60_s": p["rt60_s"], "seed": p["seed"], "damping_hz": 9000}, "hall", sr)
        w1 = _conv(x, ir)
        up = sosfilt(_butter(2, 200.0, "high", sr), np.abs(w1), axis=0)
        wet = w1 + p["amount"] * _conv(up, ir) * (_peak(w1) / _peak(up))
    elif ty == "ir":
        wet = _conv(x, _file_ir(p, sr))
    else:
        raise ValueError(f"unknown effect {ty!r}")
    return x + send * wet


def apply_chain(dry: np.ndarray, chain: list[dict], sr: int, bpm: float, space: bool = True,
                offset_s: float = 0.0) -> np.ndarray:
    """Mono dry track (n,) through a chain -> stereo (n, 2). ``space`` False gives the direct path (delays
    and reverbs muted), used to measure the wet/dry ratio."""
    x = np.stack([dry, dry], axis=1).astype(np.float64)
    t = offset_s + np.arange(len(dry)) / sr
    for n in chain:
        if "type" in n:
            x = _apply_node(x, n, sr, bpm, t if "auto" in n or n["type"] in ("tremolo", "autopan", "ringmod", "tape")
                            else None, space)
    return x


def vary_chain(chain: list[dict], kind: str, rng: np.random.Generator, n_bars: int) -> list[dict]:
    """Another setting of the same chain for another section: 1 to 3 nodes redrawn (same type, new settings),
    sometimes one node added or removed, as a producer automates or switches effects between sections."""
    head, nodes = chain[0], [dict(n) for n in chain[1:]]
    for i in rng.permutation(len(nodes))[: int(rng.integers(1, 4))]:
        new = _node(nodes[i]["type"], kind, rng)
        new["category"] = nodes[i]["category"]
        nodes[i] = new
    if rng.random() < 0.3 and nodes:
        nodes.pop(int(rng.integers(len(nodes))))
    elif rng.random() < 0.3:
        cat = str(rng.choice([c for c in CATEGORIES if c != "pan"]))
        new = _node(str(rng.choice([t for t in CATEGORIES[cat][1] if t != "ir" or _irs()])), kind, rng)
        new["category"] = cat
        nodes.insert(int(rng.integers(len(nodes) + 1)), new)
    if rng.random() < AUTO_P * 0.5:
        _automate(nodes, rng, n_bars, kind)
    return [head] + nodes


# --------------------------------------------------------------------------------------------- descriptors

def _rt60(chain: list[dict], sr: int, bpm: float) -> float | None:
    """RT60 of the chain's space effects from its impulse response (Schroeder integral, T20 x 3)."""
    if not any(n.get("type") in SPACE for n in chain[1:]):
        return None
    imp = np.zeros(int(6 * sr))
    imp[int(0.05 * sr)] = 0.5
    full = apply_chain(imp, chain, sr, bpm)
    direct = apply_chain(imp, chain, sr, bpm, space=False)
    e = np.sum((full - direct) ** 2, axis=1)
    if e.sum() <= 1e-12:
        return None
    sch = np.cumsum(e[::-1])[::-1]
    db = 10 * np.log10(sch / sch[0] + 1e-30)
    i5, i25 = np.argmax(db <= -5), np.argmax(db <= -25)
    if i25 <= i5:
        return None
    return round(float(3 * (i25 - i5) / sr), 3)


def _thd(chain: list[dict], sr: int, bpm: float, level: float) -> float:
    """Harmonic distortion of the chain's direct path on a 220 Hz tone at the source's peak level."""
    n = sr // 2
    f0 = 220.0
    tone = level * np.sin(2 * np.pi * f0 * np.arange(n) / sr)
    y = apply_chain(tone, chain, sr, bpm, space=False).mean(axis=1)[n // 4:]
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    bins = [int(np.argmin(np.abs(freqs - k * f0))) for k in range(1, 10) if k * f0 < 0.45 * sr]
    h = [float(np.max(spec[max(b - 2, 0): b + 3]) ** 2) for b in bins]
    return round(float(np.sqrt(sum(h[1:]) / (h[0] + 1e-20))), 4)


def describe(track: np.ndarray, direct: np.ndarray | None, sr: int) -> dict:
    """Descriptors measured on one rendered stretch of a source's track (n, 2), before the master: pan (energy-
    weighted level difference, -1 left .. 1 right), width (1 - |inter-channel correlation|), wet/dry ratio
    (energy of track - direct path over the direct path's, dB), spectral centroid (active frames, Hz)."""
    out = {"pan": None, "width": None, "wet_ratio_db": None, "centroid_hz": None}
    e = np.sum(track ** 2)
    if e <= 1e-12:
        return out
    left, right = track[:, 0], track[:, 1]
    el, er = np.sum(left ** 2), np.sum(right ** 2)
    out["pan"] = round(float((er - el) / (er + el + 1e-12)), 4)
    c = np.corrcoef(left, right)[0, 1] if np.std(left) > 0 and np.std(right) > 0 else 1.0
    out["width"] = round(float(1 - abs(c)), 4)
    if direct is not None:
        ed = np.sum(direct ** 2)
        out["wet_ratio_db"] = round(float(10 * np.log10((np.sum((track - direct) ** 2) + 1e-12) / (ed + 1e-12))), 2)
    m = track.mean(axis=1)
    nfft, hop = 2048, 1024
    nf = (len(m) - nfft) // hop
    if nf > 0:
        idx = np.arange(nfft)[None] + hop * np.arange(nf)[:, None]
        spec = np.abs(np.fft.rfft(m[idx] * np.hanning(nfft), axis=1))
        pw = spec.sum(axis=1)
        act = pw > 1e-3 * pw.max()
        freqs = np.fft.rfftfreq(nfft, 1 / sr)
        out["centroid_hz"] = round(float(np.sum(spec[act] * freqs) / (np.sum(spec[act]) + 1e-12)), 1)
    return out


def chain_ids(chain: list[dict]) -> list[str]:
    """The chain as a list of effect ids in order (reference only, not a training target)."""
    return [n["type"] + ("~" if "auto" in n else "") for n in chain[1:]]
