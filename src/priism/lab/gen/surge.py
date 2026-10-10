"""Tonal parts played by Surge XT with its factory and third-party patches.

The home-made synth of ``tonal.render_notes`` has one oscillator, one filter and a
few envelopes, so every bass, lead or pad sounds like a cousin of the others.
Surge XT (open source) ships about 3 000 patches made by sound designers. A part
keeps its notes (``tonal.Note``); only the instrument changes: the notes are
played into one Surge instance with a random patch of the right category, and
the result is one dry mono track like ``render_notes`` returns. FX and the master
stay those of the generator.

Needs ``surgepy`` (built from the Surge XT sources, see ``pod/setup.sh``); without
it ``available()`` is False and the generator keeps its own synth.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .tonal import Note

# Patch folders (category names in factory and third-party banks) for each kind.
CATEGORIES = {
    "bass": ("basses", "bass"),
    "lead": ("leads", "lead", "synths"),
    "pad": ("pads", "pad", "atmospheres", "ambient", "drones"),
    "pluck": ("plucks", "pluck"),
    "arp": ("plucks", "pluck", "keys"),
    "stab": ("keys", "chords", "polysynths", "polysynth", "stabs", "brass"),
}
KINDS = tuple(CATEGORIES)


def available() -> bool:
    try:
        import surgepy  # noqa: F401
    except ImportError:
        return False
    return bool(patches("bass"))


def data_path() -> Path:
    """Surge's data folder (patches): ``PRIISM_SURGE_DATA``, else ``surge-data`` next to the
    module (``pod/surge/build.sh``), else where surgepy looks for it."""
    if os.environ.get("PRIISM_SURGE_DATA"):
        return Path(os.environ["PRIISM_SURGE_DATA"])
    import surgepy

    here = Path(surgepy.__file__).parent / "surge-data"
    return here if here.is_dir() else Path(surgepy.createSurge(44100).getFactoryDataPath())


@lru_cache(maxsize=None)
def _index() -> dict[str, tuple[str, ...]]:
    root = data_path()
    out: dict[str, list[str]] = {k: [] for k in KINDS}
    for bank in ("patches_factory", "patches_3rdparty"):
        for f in sorted((root / bank).rglob("*.fxp")):
            folder = f.parent.name.lower()
            for kind, cats in CATEGORIES.items():
                if folder in cats:
                    out[kind].append(str(f))
    return {k: tuple(v) for k, v in out.items()}


def patches(kind: str) -> tuple[str, ...]:
    return _index().get(kind, ())


def sample_patch(kind: str, rng: np.random.Generator) -> str:
    pool = patches(kind)
    return pool[int(rng.integers(len(pool)))]


def render_notes(notes: list[Note], patch: str, n: int, sr: int, bpm: float) -> np.ndarray:
    """Mono track (n,) of ``notes`` played by Surge XT with ``patch`` (an .fxp path)."""
    import surgepy

    s = surgepy.createSurge(sr)
    if hasattr(s, "setTempo"):  # our build adds it; tempo-synced LFOs and delays follow the song
        s.setTempo(float(bpm))
    s.loadPatch(patch)
    bs = s.getBlockSize()
    step = 60.0 / bpm / 4.0 * sr
    events = []
    for k, note in enumerate(notes):
        pitch = int(round(note.pitch))
        if not 0 <= pitch <= 127:
            continue
        on = int(round(note.start * step))
        off = on + max(1, int(note.length * step))
        vel = int(np.clip(round(note.vel * 127), 1, 127))
        events.append((on // bs, 1, k, pitch, vel))
        events.append((off // bs, 0, k, pitch, 0))  # offs sort before ons in the same block
    events.sort()
    blocks = -(-n // bs)
    buf = s.createMultiBlock(blocks)
    held: dict[int, int] = {}  # pitch -> notes holding it (overlapping notes of one pitch)
    pos = 0
    for blk, is_on, _, pitch, vel in events:
        blk = min(blk, blocks)
        if blk > pos:
            s.processMultiBlock(buf, pos, blk - pos)
            pos = blk
        if is_on:
            if held.get(pitch):  # retrigger: Surge keeps one voice per pitch and channel
                s.releaseNote(0, pitch, 0)
            s.playNote(0, pitch, vel, 0)
            held[pitch] = held.get(pitch, 0) + 1
        elif held.get(pitch):
            held[pitch] -= 1
            if not held[pitch]:
                s.releaseNote(0, pitch, 0)
    if pos < blocks:
        s.processMultiBlock(buf, pos, blocks - pos)
    y = np.asarray(buf, dtype=np.float64).mean(0)[:n]
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
