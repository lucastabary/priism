"""Synthetic sources that training sets are built from, one per stem type.

A source is a generator of isolated, exactly known stems: TB-303 acid lines,
dub skanks, and later anything that standard models handle poorly. Adding a
stem type means writing ``sample_params`` and ``render`` for it and listing
it in ``SOURCES``; rendering, mixing over real backgrounds (``synth_mix`` in
``dataset``) and training then work the same way for every source.

Each rendered example is a stereo FLAC plus a JSON of its parameters, so a
dataset can be regenerated exactly from its seeds.
"""

from __future__ import annotations

import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Source:
    name: str
    module: str  # holds sample_params(seed, duration_s, sample_rate), params_from_dict(d) and render(params)
    description: str

    def load(self):
        return importlib.import_module(self.module)


SOURCES = {
    s.name: s
    for s in [
        Source("acid", "priism.acid.generate", "TB-303 style lines with saturation, dub delay and reverb"),
        Source("skank", "priism.skank.generate", "dub/reggae offbeat chords: guitar, organ bubble, piano, chord stabs"),
    ]
}


def get_source(name: str) -> Source:
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
    return SOURCES[name]


def render_params(source: str, params: dict) -> np.ndarray:
    """Re-render an example from the JSON written next to it."""
    mod = get_source(source).load()
    return mod.render(mod.params_from_dict(params))


def _render_one(args: tuple[str, int, float, int, str]) -> str:
    source, seed, duration_s, sample_rate, out_dir = args
    mod = get_source(source).load()
    stem = Path(out_dir) / f"{source}_{seed:07d}"
    if stem.with_suffix(".json").exists():  # written last, so the file is complete; lets a job resume
        return str(stem.with_suffix(".flac"))
    p = mod.sample_params(seed, duration_s=duration_s, sample_rate=sample_rate)
    # 24-bit FLAC: examples are peak-normalised, so this loses nothing audible at
    # about a third of the size of float WAV, which matters at 10k+ files.
    sf.write(stem.with_suffix(".flac"), mod.render(p), sample_rate, format="FLAC", subtype="PCM_24")
    stem.with_suffix(".json").write_text(json.dumps(p.to_dict(), indent=1))
    return str(stem.with_suffix(".flac"))


def generate(source: str, out_dir: str | Path, count: int, start_seed: int = 0, duration_s: float = 8.0,
             sample_rate: int = 44100, workers: int = 1) -> list[str]:
    get_source(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(source, start_seed + i, duration_s, sample_rate, str(out_dir)) for i in range(count)]
    if workers <= 1:
        return [_render_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_render_one, jobs, chunksize=4))
