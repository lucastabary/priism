"""Shared helpers for the engine tests: a synthetic track and JSON Schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src" / "priism" / "engine" / "schema"


def synth_track(seconds: float = 3.0, sr: int = 44100, seed: int = 0) -> np.ndarray:
    """(frames, 2) float in [-1, 1]: kick, bass line, chord, hats. Wideband, so band splits are all non-zero."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    beat = (t * 2) % 1.0  # 120 bpm
    kick = np.sin(2 * np.pi * 55 * t * np.exp(-beat * 3)) * np.exp(-beat * 12)
    bass = 0.4 * np.sign(np.sin(2 * np.pi * 82.4 * t)) * (beat > 0.5)
    chord = 0.15 * sum(np.sin(2 * np.pi * f * t) for f in (440.0, 554.4, 659.3))
    hats = 0.2 * rng.standard_normal(len(t)) * np.exp(-((t * 4) % 1.0) * 30)
    left = kick + bass + chord + hats
    right = kick + bass + 0.8 * chord + 0.6 * hats
    x = np.stack([left, right], 1)
    return 0.8 * x / np.max(np.abs(x))


def validate(instance: dict, schema_name: str) -> None:
    import jsonschema

    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)
