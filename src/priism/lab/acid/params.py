"""Random parameters for one synthetic acid line.

Everything that shapes a line (pattern, synth knobs, effects) lives in
``AcidParams`` so a line can be regenerated exactly from its JSON metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Semitone offsets from the root. Acid lines live mostly in minor-ish modes.
SCALES: dict[str, list[int]] = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}


@dataclass
class Step:
    """One 16th-note step of the sequencer, 303 style."""

    note: int  # MIDI note number
    gate: bool  # False is a rest
    accent: bool
    slide: bool  # glide into the next step without retriggering the envelopes


@dataclass
class SynthParams:
    waveform: str  # "saw" or "square"
    cutoff_hz: float  # filter cutoff with the envelope at rest
    resonance: float  # 0..1, self-oscillation starts close to 1
    env_mod_oct: float  # how many octaves the filter envelope opens the cutoff
    decay_s: float  # filter envelope decay time
    accent_amount: float  # 0..1, extra cutoff and level on accented steps
    glide_s: float  # slide time
    gate_fraction: float  # portion of a step the note holds when not sliding


@dataclass
class FxParams:
    drive: float  # 0 is clean; above 1 is heavy saturation
    delay_mix: float
    delay_steps_l: int  # delay time in 16th notes, left channel
    delay_steps_r: int
    delay_feedback: float
    delay_lowpass_hz: float  # darkens repeats, dub style
    reverb_mix: float
    reverb_rt60_s: float
    width: float  # 0 is mono


@dataclass
class AcidParams:
    seed: int
    sample_rate: int
    bpm: float
    root_note: int
    scale: str
    steps: list[Step]
    synth: SynthParams
    fx: FxParams
    duration_s: float
    peak_dbfs: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AcidParams":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d["steps"]]
        d["synth"] = SynthParams(**d["synth"])
        d["fx"] = FxParams(**d["fx"])
        return cls(**d)


def random_pattern(rng: np.random.Generator, root_note: int, scale: str, n_steps: int) -> list[Step]:
    degrees = SCALES[scale]
    gate_p = rng.uniform(0.55, 0.95)
    accent_p = rng.uniform(0.1, 0.4)
    slide_p = rng.uniform(0.05, 0.35)
    # Acid lines hover around the root, so the root gets extra weight.
    weights = np.ones(len(degrees))
    weights[0] = rng.uniform(2.0, 5.0)
    weights /= weights.sum()
    steps = []
    for _ in range(n_steps):
        degree = degrees[rng.choice(len(degrees), p=weights)]
        octave = rng.choice([-1, 0, 1], p=[0.15, 0.6, 0.25])
        steps.append(
            Step(
                note=int(root_note + degree + 12 * octave),
                gate=bool(rng.random() < gate_p),
                accent=bool(rng.random() < accent_p),
                slide=bool(rng.random() < slide_p),
            )
        )
    return steps


def sample_params(seed: int, duration_s: float = 8.0, sample_rate: int = 44100) -> AcidParams:
    rng = np.random.default_rng(seed)
    scale = str(rng.choice(list(SCALES), p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1]))
    root_note = int(rng.integers(28, 46))  # E1..A#2, the 303's home range
    # Patterns are 1, 2 or 4 bars of 16 steps, looped over the duration.
    n_steps = 16 * int(rng.choice([1, 2, 4], p=[0.3, 0.4, 0.3]))
    steps = random_pattern(rng, root_note, scale, n_steps)

    synth = SynthParams(
        waveform=str(rng.choice(["saw", "square"], p=[0.65, 0.35])),
        cutoff_hz=float(np.exp(rng.uniform(np.log(80), np.log(1500)))),
        resonance=float(rng.uniform(0.3, 0.97)),
        env_mod_oct=float(rng.uniform(0.5, 5.0)),
        decay_s=float(np.exp(rng.uniform(np.log(0.08), np.log(1.5)))),
        accent_amount=float(rng.uniform(0.2, 1.0)),
        glide_s=float(rng.uniform(0.03, 0.1)),
        gate_fraction=float(rng.uniform(0.4, 0.8)),
    )
    fx = FxParams(
        drive=float(rng.choice([0.0, rng.uniform(0.2, 1.0), rng.uniform(1.0, 4.0)], p=[0.3, 0.4, 0.3])),
        delay_mix=float(rng.choice([0.0, rng.uniform(0.1, 0.5)], p=[0.35, 0.65])),
        delay_steps_l=int(rng.choice([2, 3, 4, 6])),
        delay_steps_r=int(rng.choice([2, 3, 4, 6])),
        delay_feedback=float(rng.uniform(0.2, 0.75)),
        delay_lowpass_hz=float(rng.uniform(1500, 8000)),
        reverb_mix=float(rng.choice([0.0, rng.uniform(0.05, 0.35)], p=[0.4, 0.6])),
        reverb_rt60_s=float(rng.uniform(0.4, 3.5)),
        width=float(rng.uniform(0.0, 0.8)),
    )
    return AcidParams(
        seed=seed,
        sample_rate=sample_rate,
        bpm=float(rng.uniform(100, 150)),
        root_note=root_note,
        scale=scale,
        steps=steps,
        synth=synth,
        fx=fx,
        duration_s=duration_s,
        peak_dbfs=float(rng.uniform(-6.0, -1.0)),
    )
