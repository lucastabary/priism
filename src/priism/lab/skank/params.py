"""Random parameters for one synthetic skank part.

The skank is the offbeat chord chop of reggae and dub: a muted guitar
strum, the organ "bubble", a damped piano, or the filtered chord stab of
dub techno. Like the acid lines, everything that shapes a part lives in
``SkankParams`` so it can be regenerated exactly from its JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Hit positions on a 16-step bar (beats on 0, 4, 8, 12).
STYLES: dict[str, list[int]] = {
    "two_four": [4, 12],  # one drop: chops on beats 2 and 4
    "upbeats": [2, 6, 10, 14],  # every "and"
    "double": [4, 6, 12, 14],  # 2 and 4 plus the "and" after
    "chop16": [4, 5, 12, 13],  # 16th double chop
    "bubble": [2, 3, 6, 7, 10, 11, 14, 15],  # organ bubble
}
INSTRUMENTS = ["guitar", "organ", "piano", "stab"]

MINOR = [0, 2, 3, 5, 7, 8, 10]
MAJOR = [0, 2, 4, 5, 7, 9, 11]


@dataclass
class Hit:
    time_s: float
    notes: list[int]  # MIDI notes of the chord voicing
    velocity: float  # 0..1
    length_s: float  # how long the chord is held before damping
    strum_s: float  # delay between strings (guitar), signed: negative is an upstroke
    throw: bool  # sent to the echo at full level (dub mixing move)


@dataclass
class ToneParams:
    instrument: str
    damping: float  # guitar string loss per period, 0.9 (dead) .. 0.995 (ringing)
    brightness: float  # 0..1, pluck/hammer hardness
    drawbars: list[float]  # organ levels for harmonics 1, 2, 3, 4, 6, 8
    detune_cents: float  # stab oscillator spread
    highpass_hz: float  # skanks are thin: bass is cut
    lowpass_hz: float


@dataclass
class FxParams:
    drive: float
    echo_send: float  # base send to the echo; throws send 1.0
    echo_steps: int  # echo time in 16ths
    echo_feedback: float
    echo_lowpass_hz: float
    spring_mix: float
    spring_rt60_s: float
    width: float


@dataclass
class SkankParams:
    seed: int
    sample_rate: int
    bpm: float
    style: str
    hits: list[Hit]
    tone: ToneParams
    fx: FxParams
    duration_s: float
    peak_dbfs: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SkankParams":
        d = dict(d)
        d["hits"] = [Hit(**h) for h in d["hits"]]
        d["tone"] = ToneParams(**d["tone"])
        d["fx"] = FxParams(**d["fx"])
        return cls(**d)


def chord(root: int, mode: list[int], degree: int, size: int) -> list[int]:
    """Diatonic chord on ``degree`` stacked in thirds, as pitch classes above ``root``."""
    return [root + mode[(degree + 2 * k) % 7] + 12 * ((degree + 2 * k) // 7) for k in range(size)]


def voice(notes: list[int], low: int, high: int, rng: np.random.Generator) -> list[int]:
    """Fold a chord into [low, high] with a random inversion, closest to a mid register."""
    out = []
    for n in notes:
        while n < low:
            n += 12
        while n > high:
            n -= 12
        out.append(n)
    out = sorted(set(out))
    for _ in range(int(rng.integers(0, len(out)))):  # inversions
        if out[0] + 12 <= high:
            out = sorted(out[1:] + [out[0] + 12])
    return out


def sample_params(seed: int, duration_s: float = 8.0, sample_rate: int = 44100) -> SkankParams:
    rng = np.random.default_rng(seed)
    # Reggae/dub sits at 60-90 BPM; steppers and dub techno run faster.
    bpm = float(rng.choice([rng.uniform(60, 90), rng.uniform(110, 140)], p=[0.7, 0.3]))
    instrument = str(rng.choice(INSTRUMENTS, p=[0.4, 0.25, 0.15, 0.2]))
    style = str(rng.choice(list(STYLES), p=[0.3, 0.25, 0.15, 0.1, 0.2] if instrument == "organ" else [0.4, 0.3, 0.15, 0.15, 0.0]))

    mode = MINOR if rng.random() < 0.7 else MAJOR
    root = int(rng.integers(48, 60))
    n_chords = int(rng.choice([1, 2, 4], p=[0.3, 0.45, 0.25]))
    bars_per_chord = int(rng.choice([1, 2], p=[0.6, 0.4]))
    degrees = [0] + [int(rng.choice([3, 4, 5, 6, 2])) for _ in range(n_chords - 1)]
    size = int(rng.choice([3, 4], p=[0.6, 0.4]))
    low = int(rng.integers(50, 58))
    chords = [voice(chord(root, mode, d, size), low, low + 19, rng) for d in degrees]

    step_s = 60.0 / bpm / 4.0
    positions = STYLES[style]
    gate = float(rng.uniform(0.25, 0.9))  # share of the gap to the next hit the chord is held
    swing = float(rng.uniform(0.0, 0.3))
    keep_p = float(rng.uniform(0.75, 1.0))  # dub drops hits out
    throw_p = float(rng.choice([0.0, rng.uniform(0.03, 0.15)], p=[0.4, 0.6]))
    strum = float(rng.uniform(0.002, 0.012)) if instrument == "guitar" else 0.0
    hits = []
    bar_s = 16 * step_s
    for bar in range(int(np.ceil(duration_s / bar_s)) + 1):
        notes = chords[(bar // bars_per_chord) % len(chords)]
        for i, pos in enumerate(positions):
            if rng.random() > keep_p:
                continue
            t = bar * bar_s + (pos + swing * (pos % 2)) * step_s + rng.normal(0, 0.004)
            if t >= duration_s or t < 0:
                continue
            nxt = positions[i + 1] if i + 1 < len(positions) else 16 + positions[0]
            hits.append(Hit(
                time_s=float(t),
                notes=list(notes),
                velocity=float(np.clip(rng.normal(0.8, 0.1), 0.3, 1.0)),
                length_s=float(max(0.03, gate * (nxt - pos) * step_s)),
                strum_s=float(strum * (1 if rng.random() < 0.8 else -1)),
                throw=bool(rng.random() < throw_p),
            ))

    tone = ToneParams(
        instrument=instrument,
        damping=float(rng.uniform(0.93, 0.995)),
        brightness=float(rng.uniform(0.2, 1.0)),
        drawbars=[float(x) for x in rng.uniform(0, 1, 6) * np.array([1.0, 0.9, 0.7, 0.6, 0.4, 0.3])],
        detune_cents=float(rng.uniform(3, 18)),
        highpass_hz=float(np.exp(rng.uniform(np.log(120), np.log(500)))),
        lowpass_hz=float(np.exp(rng.uniform(np.log(2500), np.log(12000)))),
    )
    fx = FxParams(
        drive=float(rng.choice([0.0, rng.uniform(0.1, 0.8)], p=[0.6, 0.4])),
        echo_send=float(rng.choice([0.0, rng.uniform(0.05, 0.4)], p=[0.3, 0.7])),
        echo_steps=int(rng.choice([3, 4, 6, 8], p=[0.4, 0.3, 0.2, 0.1])),
        echo_feedback=float(rng.uniform(0.3, 0.8)),
        echo_lowpass_hz=float(rng.uniform(1200, 5000)),
        spring_mix=float(rng.choice([0.0, rng.uniform(0.1, 0.5)], p=[0.25, 0.75])),
        spring_rt60_s=float(rng.uniform(0.8, 3.0)),
        width=float(rng.uniform(0.0, 0.8)),
    )
    return SkankParams(seed=seed, sample_rate=sample_rate, bpm=bpm, style=style, hits=hits, tone=tone, fx=fx,
                       duration_s=duration_s, peak_dbfs=float(rng.uniform(-6.0, -1.0)))
