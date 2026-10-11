"""Orchestration behind the ``priism-lab`` registry commands: build the cache, pick song sets."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .features import (Cached, MsstTrunk, Projection, band_energy, load_set, read_multitrack, read_ni,
                       read_synth, song_cache, song_tokens, write_cache)

READERS = {"synth": read_synth, "ni": read_ni, "multitrack": read_multitrack}


def list_songs(kind: str, folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if kind == "synth":
        return sorted(p.parent for p in folder.glob("*/meta.json"))
    if kind == "ni":
        return sorted(folder.glob("*.stem.mp4"))
    if kind == "multitrack":
        return sorted(p for p in folder.iterdir() if p.is_dir())
    raise ValueError(f"unknown set kind {kind!r} (synth, ni, multitrack)")


def parse_sets(specs: list[str]) -> dict[str, tuple[str, Path]]:
    """``name=kind:path`` -> {name: (kind, path)}."""
    out = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        out[name] = (kind, Path(path))
    return out


def build_cache(trunk, sets: dict[str, tuple[str, Path]], out: str | Path, fit_set: str, fit_songs: int = 16,
                dims: int = 32, pool: int = 4, window_s: float = 8.0, margin_s: float = 1.0,
                limit: int | None = None, log=print) -> dict:
    """Fit the PCA on the first ``fit_songs`` of ``fit_set`` (a training set: test songs must not shape the
    features), then cache every song of every set. Existing files are kept, so a stopped run resumes.
    ``info.json`` records the seconds per song, to extrapolate the cost of larger sets."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    proj_path = out / "proj.npz"
    sr = trunk.sample_rate
    if proj_path.exists():
        proj = Projection.load(proj_path)
    else:
        kind, folder = sets[fit_set]
        proj = Projection()
        t = time.time()
        for p in list_songs(kind, folder)[:fit_songs]:
            song = READERS[kind](p, sr)
            h = song_tokens(trunk, song.mix, window_s, margin_s)
            e = band_energy(trunk, song.mix)[: len(h)]
            proj.update(h, e > e.max() * 1e-6)
        proj.fit(dims).save(proj_path)
        log(f"PCA on {fit_songs} songs of {fit_set}: {dims} dims keep {proj.explained:.1%} of the variance "
            f"({time.time() - t:.0f} s)")
    info = {"sample_rate": sr, "hop": trunk.hop, "pool": pool, "dims": dims, "window_s": window_s,
            "margin_s": margin_s, "explained": proj.explained, "frame_s": trunk.hop * pool / sr, "sets": {}}
    for name, (kind, folder) in sets.items():
        (out / name).mkdir(exist_ok=True)
        songs = list_songs(kind, folder)[:limit]
        times, audio_s = [], 0.0
        for p in songs:
            dest = out / name / f"{p.name.removesuffix('.stem.mp4')}.npz"
            if dest.exists():
                continue
            t = time.time()
            song = READERS[kind](p, sr)
            write_cache(dest, song_cache(trunk, song, proj, pool, window_s, margin_s))
            times.append(time.time() - t)
            audio_s += song.mix.shape[1] / sr
            log(f"{name}/{dest.stem}: {song.mix.shape[1] / sr:.0f} s of audio, {len(song.labels)} sources, "
                f"{times[-1]:.1f} s")
        info["sets"][name] = {"kind": kind, "songs": len(songs), "new": len(times),
                              "seconds": float(np.sum(times)), "audio_seconds": audio_s}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    return info


def select(cache: str | Path, spec: str) -> list[Cached]:
    """``set`` or ``set%dev`` / ``set%test``: the two halves of a set, split by a hash of the song name
    (stable whatever is added later), so thresholds can be tuned on one half of real songs and scored on
    the other. ``set%train`` / ``set%dev2`` split the dev half again (about 2/3 and 1/3, another hash), to
    train on real songs while ``%test`` stays exactly the half that phase 1 was scored on."""
    name, _, part = spec.partition("%")
    songs = load_set(Path(cache) / name)
    return [s for s in songs if in_part(s.name, part)] if part else songs


def in_part(song: str, part: str) -> bool:
    """Whether a song name falls in ``part`` (dev, test, train, dev2); usable without loading the cache."""
    if part not in ("dev", "test", "train", "dev2"):
        raise ValueError(f"unknown part {part!r}: dev, test, train or dev2")
    half = "dev" if int(hashlib.md5(song.encode()).hexdigest(), 16) % 2 == 0 else "test"
    if part in ("dev", "test"):
        return half == part
    if half != "dev":
        return False
    third = int(hashlib.sha1(song.encode()).hexdigest(), 16) % 3
    return (third != 0) == (part == "train")

