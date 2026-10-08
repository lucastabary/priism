"""Tonal parts (basses, synths, chords, FX) that follow the song's tempo, key and chords.

Every part is a list of note events played by a randomly drawn patch. The
patch is a small subtractive/FM synth; the 303 and the skank reuse the
existing ``priism.acid`` and ``priism.skank`` voices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

from ..acid.params import AcidParams, FxParams as AcidFx, SynthParams as AcidSynth, random_pattern
from ..acid.synth import ladder_filter, midi_to_hz, oscillator, render_dry as acid_render_dry
from ..dsp import saturate
from ..skank.params import STYLES as SKANK_STYLES, Hit, ToneParams
from ..skank.synth import render_hit as skank_hit

SCALES: dict[str, list[int]] = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor_pentatonic": [0, 3, 5, 7, 10],
}


@dataclass
class Harmony:
    """Key and chord loop shared by all tonal parts of a song."""

    root: int  # pitch class 0..11 of the tonic
    mode: str
    degrees: list[int]  # chord loop, scale degrees
    bars_per_chord: int

    def scale(self) -> list[int]:
        return SCALES[self.mode]

    def chord_degree(self, bar: int) -> int:
        return self.degrees[(bar // self.bars_per_chord) % len(self.degrees)]

    def note(self, degree: int, octave: int) -> int:
        """MIDI note of a scale degree (may exceed the scale length) in a given octave (C4 = octave 4)."""
        sc = self.scale()
        return 12 * (octave + 1) + self.root + sc[degree % len(sc)] + 12 * (degree // len(sc))

    def chord(self, bar: int, octave: int, size: int = 3) -> list[int]:
        d = self.chord_degree(bar)
        step = 2 if len(self.scale()) == 7 else 1
        return [self.note(d + step * k, octave) for k in range(size)]


@dataclass
class Note:
    start: float  # in 16th steps from the song start
    length: float  # in 16th steps
    pitch: float  # MIDI, may be fractional
    vel: float


# ---------------------------------------------------------------------------------------------- patches

def sample_patch(kind: str, rng: np.random.Generator) -> dict:
    u = rng.uniform
    osc_p = {
        "sub": (["sine", "tri"], [0.7, 0.3]),
        "bass": (["saw", "square", "fm", "sine", "tri", "supersaw"], [0.3, 0.2, 0.15, 0.1, 0.1, 0.15]),
        "stab": (["saw", "supersaw", "square", "fm"], [0.35, 0.35, 0.15, 0.15]),
        "pad": (["supersaw", "saw", "tri", "fm"], [0.45, 0.25, 0.15, 0.15]),
        "lead": (["saw", "square", "supersaw", "fm", "tri"], [0.3, 0.25, 0.2, 0.15, 0.1]),
        "arp": (["saw", "square", "fm", "tri"], [0.3, 0.3, 0.2, 0.2]),
        "pluck": (["saw", "square", "fm", "sine"], [0.3, 0.25, 0.25, 0.2]),
    }[kind]
    osc = str(rng.choice(osc_p[0], p=osc_p[1]))
    p = {
        "osc": osc,
        "voices": int(rng.integers(3, 8)) if osc == "supersaw" else 1,
        "detune_cents": u(5, 35),
        "fm_ratio": float(rng.choice([0.5, 1.0, 2.0, 3.0, 1.41, 7.0])),
        "fm_index": u(0.5, 6.0),
        "fm_decay": u(0.05, 1.0),
        "sub": u(0, 0.5) if kind in ("bass", "lead") else 0.0,
        "noise": u(0, 0.05),
        "cutoff": float(np.exp(u(np.log(150), np.log(6000)))),
        "res": u(0.0, 0.8),
        "env_oct": u(0.0, 4.0),
        "f_decay": float(np.exp(u(np.log(0.03), np.log(1.0)))),
        "attack": u(0.001, 0.01),
        "decay": u(0.05, 0.6),
        "sustain": u(0.3, 1.0),
        "release": u(0.01, 0.2),
        "drive": float(rng.choice([0.0, u(0.1, 1.5)], p=[0.5, 0.5])),
        "lfo_steps": float(rng.choice([1, 2, 4, 8, 16])),  # filter LFO period in 16ths
        "lfo_oct": 0.0,
    }
    if kind == "sub":
        p.update(cutoff=u(100, 400), res=0.0, env_oct=u(0, 1), sustain=1.0, release=u(0.02, 0.1), drive=u(0, 0.4))
    elif kind == "bass":
        p.update(cutoff=float(np.exp(u(np.log(120), np.log(2000)))))
    elif kind == "stab":
        p.update(sustain=u(0.0, 0.3), decay=u(0.08, 0.5), env_oct=u(0.5, 3.0), f_decay=u(0.03, 0.3))
    elif kind == "pad":
        p.update(attack=u(0.2, 2.0), decay=u(0.5, 2.0), sustain=u(0.6, 1.0), release=u(0.5, 2.5), env_oct=u(0, 1),
                 lfo_oct=float(rng.choice([0.0, u(0.2, 1.5)])), lfo_steps=float(rng.choice([16, 32, 64])))
    elif kind == "pluck":
        p.update(sustain=0.0, decay=u(0.05, 0.4), env_oct=u(1, 4), f_decay=u(0.02, 0.2), release=u(0.02, 0.1))
    return p


def _osc(p: dict, f: float, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    freq = np.full(n, f)
    o = p["osc"]
    if o in ("saw", "square"):
        y = oscillator(freq, sr, o)
    elif o == "sine":
        y = np.sin(2 * np.pi * np.cumsum(freq) / sr + rng.uniform(0, 2 * np.pi))
    elif o == "tri":
        ph = (np.cumsum(freq) / sr + rng.uniform()) % 1.0
        y = 4.0 * np.abs(ph - 0.5) - 1.0
    elif o == "supersaw":
        k = p["voices"]
        spreads = np.linspace(-1, 1, k) * p["detune_cents"] / 1200
        y = sum(oscillator(freq * 2.0**s, sr, "saw") for s in spreads) / np.sqrt(k)
    elif o == "fm":
        t = np.arange(n) / sr
        index = p["fm_index"] * np.exp(-t / p["fm_decay"])
        mod = np.sin(2 * np.pi * f * p["fm_ratio"] * t)
        y = np.sin(2 * np.pi * f * t + index * mod)
    else:
        raise KeyError(o)
    if p["sub"]:
        y = y + p["sub"] * np.sin(2 * np.pi * np.cumsum(freq / 2) / sr)
    if p["noise"]:
        y = y + p["noise"] * rng.uniform(-1, 1, n)
    return y


def _adsr(p: dict, n_hold: int, n: int, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    a, d, s, r = p["attack"], p["decay"], p["sustain"], p["release"]
    env = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / max(d, 1e-3)))
    hold_t = n_hold / sr
    level_at_release = env[min(n_hold, n - 1)] if n_hold < n else env[-1]
    rel = level_at_release * np.exp(-(t - hold_t) / max(r, 1e-3) * 3)
    return np.where(t < hold_t, env, rel)


def render_notes(notes: list[Note], p: dict, n: int, sr: int, bpm: float, rng: np.random.Generator) -> np.ndarray:
    step_s = 60.0 / bpm / 4.0
    y = np.zeros(n)
    for note in notes:
        a = int(round(note.start * step_s * sr))
        if a >= n:
            continue
        hold = max(1, int(note.length * step_s * sr))
        m = min(hold + int(p["release"] * 3 * sr) + 1, n - a)
        if m <= 0:
            continue
        f = float(midi_to_hz(note.pitch))
        x = _osc(p, f, m, sr, rng) * _adsr(p, hold, m, sr)
        t = np.arange(m) / sr
        octs = p["env_oct"] * np.exp(-t / p["f_decay"])
        if p["lfo_oct"]:
            gt = (a + np.arange(m)) / sr  # LFO locked to the song clock, not the note
            octs = octs + p["lfo_oct"] * 0.5 * (1 - np.cos(2 * np.pi * gt / (p["lfo_steps"] * step_s)))
        cutoff = np.clip(p["cutoff"] * 2.0**octs, 30, 0.42 * sr)
        x = ladder_filter(x, cutoff, p["res"], sr)
        y[a:a + m] += note.vel * x
    return saturate(y / (np.max(np.abs(y)) + 1e-9), p["drive"]) if p["drive"] else y


# ---------------------------------------------------------------------------------------------- parts

def _bars_notes(h: Harmony, bars: list[int], rng: np.random.Generator, pattern: list[tuple[int, float, int]],
                octave: int, vel_jitter: float = 0.08) -> list[Note]:
    """Place a one-bar (step, length, degree offset from chord root) pattern on every bar, following the chords."""
    out = []
    for bar in bars:
        root_deg = h.chord_degree(bar)
        for step, length, off in pattern:
            out.append(Note(bar * 16 + step, length, h.note(root_deg + off, octave),
                            float(np.clip(rng.normal(0.85, vel_jitter), 0.3, 1.0))))
    return out


BASS_STYLES = ["offbeat", "house", "electro", "rolling", "reese", "dub", "wobble"]


def bass_notes(style: str, h: Harmony, bars: list[int], rng: np.random.Generator) -> tuple[list[Note], dict]:
    octave = int(rng.choice([1, 2], p=[0.5, 0.5]))
    patch_over: dict = {}
    if style == "offbeat":
        L = float(rng.uniform(0.6, 1.8))
        pat = [(s, L, 0) for s in [2, 6, 10, 14]]
        if rng.random() < 0.5:
            pat += [(s, L, int(rng.choice([0, 7, 4]))) for s in [3, 11] if rng.random() < 0.6]
    elif style == "house":
        steps = sorted(set([0] + [int(s) for s in rng.choice([3, 6, 7, 10, 11, 13, 14], int(rng.integers(2, 5)), replace=False)]))
        pat = [(s, float(rng.uniform(1, 2.5)), int(rng.choice([0, 0, 4, 7]))) for s in steps]
    elif style == "electro":
        steps = [s for s in range(16) if rng.random() < 0.45] or [0]
        pat = [(s, 0.9, int(rng.choice([0, 0, 0, 2, 4, 7, -3]))) for s in steps]
    elif style == "rolling":
        pat = [(s, 0.8, int(rng.choice([0, 0, 0, 7, 4]))) for s in range(16) if rng.random() < 0.8]
    elif style == "reese":
        pat = [(0, 16.0, 0)] if rng.random() < 0.6 else [(0, 6.0, 0), (8, 6.0, int(rng.choice([0, 2, 4])))]
        patch_over = {"osc": "supersaw", "voices": int(rng.integers(2, 5)), "detune_cents": float(rng.uniform(10, 40)),
                      "attack": 0.01, "sustain": 1.0, "lfo_oct": float(rng.uniform(0.3, 1.5)),
                      "lfo_steps": float(rng.choice([4, 8, 16, 32]))}
    elif style == "dub":
        # Two-bar melodic phrase on chord tones, with rests: the dub bassline is the riddim's melody.
        pat2 = []
        t = 0.0
        while t < 32:
            length = float(rng.choice([2, 3, 4, 6, 8], p=[0.35, 0.15, 0.3, 0.1, 0.1]))
            if rng.random() < 0.75:
                pat2.append((t, length * 0.85, int(rng.choice([0, 2, 4, 5, 7, -1, -3]))))
            t += length
        out = []
        for bar in bars:
            root_deg = h.chord_degree(bar)
            for s, length, off in pat2:
                if int(s // 16) == bar % 2:
                    out.append(Note(bar * 16 + s % 16, length, h.note(root_deg + off, octave),
                                    float(np.clip(rng.normal(0.85, 0.08), 0.4, 1.0))))
        return out, {"osc": str(rng.choice(["sine", "tri"])), "cutoff": float(rng.uniform(200, 700)), "res": 0.1,
                     "attack": 0.005, "sustain": 0.9, "env_oct": 0.5}
    elif style == "wobble":
        pat = [(0, 8.0, 0), (8, 8.0, int(rng.choice([0, 0, -2, 3])))]
        patch_over = {"osc": str(rng.choice(["saw", "supersaw", "square"])), "sustain": 1.0, "attack": 0.005,
                      "lfo_oct": float(rng.uniform(2.0, 4.5)), "lfo_steps": float(rng.choice([1, 2, 4, 8 / 3])),
                      "cutoff": float(rng.uniform(80, 250)), "res": float(rng.uniform(0.3, 0.8))}
    else:
        raise KeyError(style)
    return _bars_notes(h, bars, rng, pat, octave), patch_over


def sub_notes(h: Harmony, bars: list[int], rng: np.random.Generator) -> list[Note]:
    if rng.random() < 0.5:
        pat = [(0, 15.5, 0)]
    else:
        pat = [(s, 3.5, 0) for s in (0, 4, 8, 12)] if rng.random() < 0.5 else [(0, 7.5, 0), (10, 5.5, 0)]
    return _bars_notes(h, bars, rng, pat, 1)


def chord_notes(kind: str, h: Harmony, bars: list[int], rng: np.random.Generator) -> list[Note]:
    octave = int(rng.choice([3, 4], p=[0.6, 0.4]))
    size = int(rng.choice([3, 4], p=[0.6, 0.4]))
    out = []
    if kind == "pad":
        for bar in bars:
            if bar % h.bars_per_chord == 0 or bar == bars[0]:
                length = 16 * (h.bars_per_chord - bar % h.bars_per_chord)
                for p in h.chord(bar, octave, size):
                    out.append(Note(bar * 16, length - 0.5, p, 0.7))
        return out
    # Stabs: a one-bar rhythm of short chords.
    style = rng.choice(["offbeat", "dotted", "syncopated", "one"])
    steps = {"offbeat": [2, 6, 10, 14], "dotted": [0, 3, 6, 10, 13], "one": [0] if rng.random() < 0.5 else [14],
             "syncopated": sorted(set(int(s) for s in rng.choice(range(16), int(rng.integers(2, 5)), replace=False)))}[style]
    length = float(rng.uniform(0.5, 2.0))
    for bar in bars:
        for s in steps:
            for p in h.chord(bar, octave, size):
                out.append(Note(bar * 16 + s, length, p, float(np.clip(rng.normal(0.8, 0.07), 0.4, 1.0))))
    return out


def melody_notes(kind: str, h: Harmony, bars: list[int], rng: np.random.Generator) -> list[Note]:
    octave = int(rng.choice([4, 5]))
    sc = h.scale()
    if kind == "arp":
        rate = float(rng.choice([1, 2], p=[0.6, 0.4]))
        span = int(rng.choice([1, 2]))
        shape = rng.choice(["up", "down", "updown", "random"])
        out = []
        for bar in bars:
            tones = []
            for o in range(span):
                tones += h.chord(bar, octave + o, 3)
            if shape == "down":
                tones = tones[::-1]
            elif shape == "updown":
                tones = tones + tones[-2:0:-1]
            n_per_bar = int(16 / rate)
            for i in range(n_per_bar):
                p = tones[int(rng.integers(len(tones)))] if shape == "random" else tones[i % len(tones)]
                out.append(Note(bar * 16 + i * rate, rate * 0.8, p, float(np.clip(rng.normal(0.8, 0.08), 0.4, 1.0))))
        return out
    # Lead or pluck: a 2- or 4-bar motif repeated, notes drawn around the chord.
    motif_bars = int(rng.choice([1, 2, 4], p=[0.2, 0.5, 0.3]))
    motif = []
    t = 0.0
    dens = rng.uniform(0.4, 0.85) if kind == "lead" else rng.uniform(0.3, 0.7)
    while t < 16 * motif_bars:
        length = float(rng.choice([1, 2, 3, 4, 6, 8], p=[0.2, 0.3, 0.1, 0.2, 0.1, 0.1]))
        if rng.random() < dens:
            motif.append((t, length, int(rng.integers(-2, len(sc) + 2))))
        t += length
    if not motif:
        motif = [(0.0, 4.0, 0)]
    out = []
    for bar in bars:
        root_deg = h.chord_degree(bar)
        for s, length, off in motif:
            if int(s // 16) == bar % motif_bars:
                out.append(Note(bar * 16 + s % 16, length * 0.9, h.note(root_deg + off, octave),
                                float(np.clip(rng.normal(0.8, 0.08), 0.4, 1.0))))
    return out


# ---------------------------------------------------------------------------------------------- special voices

def render_acid(h: Harmony, role: str, n: int, sr: int, bpm: float, rng: np.random.Generator, seed: int,
                synth: dict | None = None, octave_shift: int = 0) -> tuple[np.ndarray, dict]:
    """A 303 line in the song's key. ``role`` 'rhythmic' is a short root-heavy loop, 'melodic' a long phrase."""
    acid_scale = {"major": "dorian"}.get(h.mode, h.mode)
    root = 24 + h.root + 12 * octave_shift  # C1 octave, the 303's home is E1..A#2
    if root < 28:
        root += 12
    if role == "rhythmic":
        steps = random_pattern(rng, root, acid_scale, 16)
        for s in steps:
            if rng.random() < 0.7:
                s.note = root + 12 * int(rng.choice([0, 0, 1, -1]) if root >= 36 else rng.choice([0, 0, 1]))
            s.gate = s.gate or rng.random() < 0.5
    else:
        steps = random_pattern(rng, root, acid_scale, int(rng.choice([32, 64])))
        sc = SCALES[acid_scale]
        for s in steps:  # wider melodic range
            if rng.random() < 0.4:
                s.note = root + sc[int(rng.integers(len(sc)))] + 12 * int(rng.choice([0, 1, 1, 2]))
    if synth is None:
        synth = {
            "waveform": str(rng.choice(["saw", "square"], p=[0.65, 0.35])),
            "cutoff_hz": float(np.exp(rng.uniform(np.log(80), np.log(1500)))),
            "resonance": float(rng.uniform(0.3, 0.97)),
            "env_mod_oct": float(rng.uniform(0.5, 5.0)),
            "decay_s": float(np.exp(rng.uniform(np.log(0.08), np.log(1.5)))),
            "accent_amount": float(rng.uniform(0.2, 1.0)),
            "glide_s": float(rng.uniform(0.03, 0.1)),
            "gate_fraction": float(rng.uniform(0.4, 0.8)),
        }
    p = AcidParams(seed=seed, sample_rate=sr, bpm=bpm, root_note=root, scale=acid_scale, steps=steps,
                   synth=AcidSynth(**synth), fx=AcidFx(0, 0, 2, 2, 0, 5000, 0, 1, 0), duration_s=n / sr,
                   peak_dbfs=-1.0)
    y = acid_render_dry(p).astype(np.float64)[:n]
    return np.pad(y, (0, n - len(y))), {"synth": synth, "root": root, "steps": len(steps)}


def render_skank(h: Harmony, bars: list[int], n: int, sr: int, bpm: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    instrument = str(rng.choice(["guitar", "organ", "piano", "stab"], p=[0.4, 0.25, 0.15, 0.2]))
    style = str(rng.choice(list(SKANK_STYLES), p=[0.3, 0.25, 0.15, 0.1, 0.2] if instrument == "organ"
                           else [0.4, 0.3, 0.15, 0.15, 0.0]))
    tone = ToneParams(
        instrument=instrument, damping=float(rng.uniform(0.93, 0.995)), brightness=float(rng.uniform(0.2, 1.0)),
        drawbars=[float(x) for x in rng.uniform(0, 1, 6) * np.array([1.0, 0.9, 0.7, 0.6, 0.4, 0.3])],
        detune_cents=float(rng.uniform(3, 18)), highpass_hz=float(np.exp(rng.uniform(np.log(120), np.log(500)))),
        lowpass_hz=float(np.exp(rng.uniform(np.log(2500), np.log(12000)))))
    step_s = 60.0 / bpm / 4.0
    gate = float(rng.uniform(0.25, 0.9))
    strum = float(rng.uniform(0.002, 0.012)) if instrument == "guitar" else 0.0
    positions = SKANK_STYLES[style]
    y = np.zeros(n)
    for bar in bars:
        notes = h.chord(bar, 4, int(rng.choice([3, 4])))
        for i, pos in enumerate(positions):
            if rng.random() < 0.05:
                continue
            t = (bar * 16 + pos) * step_s + rng.normal(0, 0.004)
            nxt = positions[i + 1] if i + 1 < len(positions) else 16 + positions[0]
            hit = Hit(time_s=t, notes=list(notes), velocity=float(np.clip(rng.normal(0.8, 0.1), 0.3, 1.0)),
                      length_s=max(0.03, gate * (nxt - pos) * step_s), strum_s=strum, throw=False)
            x = skank_hit(hit, tone, sr, rng)
            a = int(round(max(t, 0) * sr))
            m = min(len(x), n - a)
            if m > 0:
                y[a:a + m] += x[:m]
    sos = butter(2, [tone.highpass_hz, min(tone.lowpass_hz, 0.45 * sr)], btype="band", fs=sr, output="sos")
    return sosfilt(sos, y), {"instrument": instrument, "style": style}


def render_siren(bars: list[int], n: int, sr: int, bpm: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Dub siren: an oscillator whose pitch is swept by a fast LFO, fired for a bar or two now and then."""
    step_s = 60.0 / bpm / 4.0
    y = np.zeros(n)
    base = float(rng.uniform(300, 900))
    depth = float(rng.uniform(0.3, 1.5))  # octaves
    rate = float(rng.uniform(1.5, 9.0))
    wave = str(rng.choice(["sine", "square", "tri"]))
    fires = [b for b in bars if rng.random() < 0.3] or bars[:1]
    for b in fires:
        length = int(rng.choice([1, 2]))
        a = int(b * 16 * step_s * sr)
        m = min(int(length * 16 * step_s * sr), n - a)
        if m <= 0:
            continue
        t = np.arange(m) / sr
        f = base * 2.0 ** (depth * np.sin(2 * np.pi * rate * t) + float(rng.uniform(-0.5, 0.5)) * t / t[-1])
        ph = np.cumsum(f) / sr
        x = np.sin(2 * np.pi * ph) if wave == "sine" else (np.sign(np.sin(2 * np.pi * ph)) * 0.5 if wave == "square"
                                                           else 4 * np.abs(ph % 1 - 0.5) - 1)
        env = np.minimum(1, t / 0.02) * np.minimum(1, (t[-1] - t) / 0.05)
        y[a:a + m] += x * env
    return y, {"base_hz": base, "depth_oct": depth, "rate_hz": rate, "wave": wave}


def render_noise_fx(bars: list[int], section_starts: list[int], n: int, sr: int, bpm: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Risers before section changes and occasional downlifters: filtered noise sweeps."""
    step_s = 60.0 / bpm / 4.0
    y = np.zeros(n)
    bar_set = set(bars)
    events = 0
    for s in section_starts:
        length = int(rng.choice([2, 4, 8]))
        if s - length < 0 or (s - 1) not in bar_set:
            continue
        a = int((s - length) * 16 * step_s * sr)
        m = min(int(length * 16 * step_s * sr), n - a)
        if m <= 0:
            continue
        x = rng.uniform(-1, 1, m)
        # Rising band-pass sweep, done in blocks.
        out = np.zeros(m)
        blocks = 32
        for i in range(blocks):
            lo, hi = i * m // blocks, (i + 1) * m // blocks
            fc = 200 * 2 ** (6 * i / blocks)
            out[lo:hi] = sosfilt(butter(2, [fc * 0.7, min(fc * 1.4, 0.45 * sr)], btype="band", fs=sr, output="sos"), x[lo:hi])
        y[a:a + m] += out * np.linspace(0.05, 1, m) ** 2
        events += 1
    if events == 0 and bars:  # a downlifter so the track is not empty
        a = int(bars[0] * 16 * step_s * sr)
        m = min(int(2 * 16 * step_s * sr), n - a)
        y[a:a + m] += rng.uniform(-1, 1, m) * np.exp(-np.arange(m) / sr / 0.8)
    return y, {"events": events}
