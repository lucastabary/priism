"""Training data in the layout Music-Source-Separation-Training reads.

Two steps:

* ``restem``: turn multitrack folders (MUSDB18-HQ, MoisesDB exports, or
  tracks pseudo-labelled with ``priism separate``) into folders holding one
  file per slot, through the same slot mapping as the separation profiles.
  These full tracks have no acid and teach the model to keep drums, bass and
  the rest right (no catastrophic forgetting).
* ``acid_mix``: cut random chunks of those slot folders and lay a synthetic
  acid line over most of them. The acid target is the synthetic line, the
  other slots stay what they were, so ground truth is exact. Some chunks get
  no acid on purpose, so the model learns to leave the acid slot empty.

Both write MSST "type 4" folders (aligned stems, mixture = sum of stems).
Train with ``audio.min_mean_abs: 0`` so silent acid targets are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .separate import Profile, assemble_slots, read_audio

SR = 44100
FORMATS = {"flac": {"format": "FLAC", "subtype": "PCM_24"}, "wav": {"format": "WAV", "subtype": "FLOAT"}}


def _write(path: Path, audio: np.ndarray, fmt: str = "flac") -> None:
    """FLAC keeps training sets small; validation sets use WAV, which MSST validation expects."""
    sf.write(path.with_suffix(f".{fmt}"), np.clip(audio, -1.0, 1.0), SR, **FORMATS[fmt])


def restem(src_dirs: list[str | Path], out_dir: str | Path, slots: dict[str, list[str] | str],
           overwrite: bool = False, with_mixture: bool = False, fmt: str = "flac") -> list[Path]:
    """Map every multitrack folder under ``src_dirs`` onto ``slots``.

    A folder is a multitrack track when it holds ``<stem>.wav`` or
    ``<stem>.flac`` files for the stems the mapping needs. A residual slot
    takes ``mixture.*`` minus the other slots when a mixture exists, else
    the sum of the stems no other slot uses.
    """
    out_dir = Path(out_dir)
    profile = Profile("restem", "-", slots)
    used = {s for v in slots.values() if v != "residual" for s in v}
    written = []
    for src in map(Path, src_dirs):
        for track in sorted(p for p in src.iterdir() if p.is_dir()):
            files = {f.stem.lower(): f for f in track.iterdir() if f.suffix.lower() in (".wav", ".flac")}
            if not used <= files.keys():
                continue
            dest = out_dir / f"{src.name}__{track.name}"
            if dest.exists() and not overwrite:
                written.append(dest)
                continue
            stems = {k: read_audio(f, SR) for k, f in files.items() if k != "mixture"}
            n = max(len(x) for x in stems.values())
            stems = {k: np.pad(x, ((0, n - len(x)), (0, 0))) for k, x in stems.items()}
            mix = read_audio(files["mixture"], SR)[:n] if "mixture" in files else sum(stems.values())
            mix = np.pad(mix, ((0, n - len(mix)), (0, 0)))
            dest.mkdir(parents=True, exist_ok=True)
            for slot, audio in assemble_slots(mix, stems, profile).items():
                _write(dest / slot, audio, fmt)
            if with_mixture:
                _write(dest / "mixture", mix, fmt)
            written.append(dest)
    return written


@dataclass
class MixSettings:
    chunk_s: float = 13.35  # BS-Roformer-SW trains on 588800-sample chunks
    acid_prob: float = 0.8
    acid_rel_db: tuple[float, float] = (-14.0, 2.0)  # acid RMS relative to the background
    min_background_rms: float = 0.005


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _load_slots(folder: Path, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        f = next((folder / f"{name}{ext}" for ext in (".flac", ".wav") if (folder / f"{name}{ext}").exists()), None)
        out[name] = sf.read(f, dtype="float32", always_2d=True)[0] if f else None
    n = max(len(x) for x in out.values() if x is not None)
    return {k: (np.zeros((n, 2), np.float32) if v is None else v) for k, v in out.items()}


def acid_mix(background_dirs: list[str | Path], acid_dir: str | Path, out_dir: str | Path, count: int,
             slots: list[str] = ("drums", "bass", "acid", "rest"), acid_slot: str = "acid",
             seed: int = 0, settings: MixSettings | None = None, start_index: int = 0,
             with_mixture: bool = False, fmt: str = "flac") -> list[Path]:
    """Write ``count`` chunk folders of real backgrounds with synthetic acid on top.

    ``with_mixture`` also writes ``mixture.flac``, which MSST validation needs.
    """
    s = settings or MixSettings()
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    backgrounds = sorted(p for d in map(Path, background_dirs) for p in d.iterdir() if p.is_dir())
    acids = sorted(p for p in Path(acid_dir).iterdir() if p.suffix in (".flac", ".wav"))
    if not backgrounds or not acids:
        raise ValueError(f"need backgrounds ({len(backgrounds)}) and acid lines ({len(acids)})")
    n = int(round(s.chunk_s * SR))
    bg_slots = [x for x in slots if x != acid_slot]
    written = []
    cache: dict[Path, dict[str, np.ndarray]] = {}
    i = start_index
    attempts = 0
    while len(written) < count:
        attempts += 1
        if attempts > 20 * count + 100:
            raise RuntimeError("backgrounds are too quiet or too short to fill the dataset")
        bg_path = backgrounds[rng.integers(len(backgrounds))]
        if bg_path not in cache:
            if len(cache) > 64:  # keep memory bounded on large datasets
                cache.pop(next(iter(cache)))
            cache[bg_path] = _load_slots(bg_path, bg_slots)
        bg = cache[bg_path]
        length = len(next(iter(bg.values())))
        if length < n:
            continue
        off = int(rng.integers(length - n + 1))
        chunk = {k: v[off : off + n] for k, v in bg.items()}
        bg_rms = _rms(sum(chunk.values()))
        if bg_rms < s.min_background_rms:
            continue

        acid = np.zeros((n, 2), np.float32)
        meta = {"background": str(bg_path), "offset": off, "acid": None}
        if rng.random() < s.acid_prob:
            acid_path = acids[rng.integers(len(acids))]
            line, _ = sf.read(acid_path, dtype="float32", always_2d=True)
            if len(line) < n:  # loop short lines to fill the chunk
                line = np.tile(line, (int(np.ceil(n / len(line))), 1))
            a_off = int(rng.integers(len(line) - n + 1))
            acid = line[a_off : a_off + n]
            rel_db = rng.uniform(*s.acid_rel_db)
            acid = acid * (bg_rms * 10 ** (rel_db / 20) / _rms(acid))
            meta["acid"] = {"file": str(acid_path), "offset": a_off, "rel_db": rel_db}

        stems = {**chunk, acid_slot: acid}
        mix = sum(stems.values())
        peak = float(np.max(np.abs(mix)))
        gain = min(1.0, 0.98 / peak) * 10 ** (rng.uniform(-6, 0) / 20)
        if rng.random() < 0.5:  # channel swap
            stems = {k: v[:, ::-1] for k, v in stems.items()}

        dest = out_dir / f"mix_{i:07d}"
        dest.mkdir(parents=True, exist_ok=True)
        for slot in slots:
            _write(dest / slot, stems[slot] * gain, fmt)
        if with_mixture:
            _write(dest / "mixture", sum(stems.values()) * gain, fmt)
        meta["gain"] = gain
        (dest / "meta.json").write_text(json.dumps(meta))
        written.append(dest)
        i += 1
    return written
