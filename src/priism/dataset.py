"""Training data in the layout Music-Source-Separation-Training reads.

Two steps:

* ``restem``: turn multitrack folders (MUSDB18-HQ, MoisesDB exports, or
  tracks pseudo-labelled with ``priism separate``) into folders holding one
  file per slot, through the same slot mapping as the separation profiles.
  These full tracks have no acid and teach the model to keep drums, bass and
  the rest right (no catastrophic forgetting).
* ``synth_mix``: cut random chunks of those slot folders and lay synthetic
  examples (acid lines, skanks, any source of ``priism.sources``) over most
  of them. A layer's target is the synthetic audio, the other slots stay
  what they were, so ground truth is exact. Some chunks get no layer on
  purpose, so the model learns to leave a slot empty. ``acid_mix`` is the
  single-layer case the acid-v1 run was built with.

Both write MSST "type 4" folders (aligned stems, mixture = sum of stems).
Train with ``audio.min_mean_abs: 0`` so silent targets are kept.
"""

from __future__ import annotations

import json
import os
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
    # Validation sets: skip chunks where a background slot is (nearly) silent, since
    # SDR of a silent target is meaningless (it reads -100 dB and wrecks averages).
    min_slot_rms: float = 0.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _load_slots(folder: Path, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        f = next((folder / f"{name}{ext}" for ext in (".flac", ".wav") if (folder / f"{name}{ext}").exists()), None)
        out[name] = sf.read(f, dtype="float32", always_2d=True)[0] if f else None
    n = max(len(x) for x in out.values() if x is not None)
    return {k: (np.zeros((n, 2), np.float32) if v is None else v) for k, v in out.items()}


@dataclass
class Layer:
    """Synthetic examples from ``dir`` laid over the background into ``slot``.

    ``slot`` can be a new stem (``acid``, ``skank``) or a background slot: a
    layer into ``rest`` is a distractor, e.g. synthetic acid lines a skank
    model must learn to leave alone. Its level is drawn relative to the
    background RMS, and it is present in a share ``prob`` of the examples.
    """

    slot: str
    dir: str | Path
    prob: float = 0.8
    rel_db: tuple[float, float] = (-14.0, 2.0)
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or self.slot


def parse_layer(spec: str) -> Layer:
    """``slot=dir[:prob[:min_db:max_db]]``, e.g. ``skank=data/skank:0.8`` or ``rest=data/acid:0.3:-20:-6``."""
    slot, _, rest = spec.partition("=")
    if not slot or not rest:
        raise ValueError(f"layer {spec!r}: expected slot=dir[:prob[:min_db:max_db]]")
    parts = rest.split(":")
    layer = Layer(slot=slot, dir=parts[0])
    if len(parts) > 1:
        layer.prob = float(parts[1])
    if len(parts) == 4:
        layer.rel_db = (float(parts[2]), float(parts[3]))
    elif len(parts) not in (1, 2):
        raise ValueError(f"layer {spec!r}: give both min_db and max_db")
    return layer


def synth_mix(background_dirs: list[str | Path], layers: list[Layer], out_dir: str | Path, count: int,
              slots: list[str], seed: int = 0, settings: MixSettings | None = None, start_index: int = 0,
              with_mixture: bool = False, fmt: str = "flac", cache_size: int = 64,
              fold_into: str | None = None) -> list[Path]:
    """Write ``count`` chunk folders of real backgrounds with synthetic layers on top.

    Background slots are read from the restem folders (a slot missing there is
    silent); each layer adds a random excerpt of one of its examples to its
    slot. ``with_mixture`` also writes ``mixture.*``, which MSST validation
    needs. ``cache_size`` bounds how many decoded backgrounds stay in memory
    (~250 MB each for a MUSDB track); lower it when several processes run
    side by side. ``fold_into`` adds the background slots that are not in
    ``slots`` to that slot, e.g. ``slots=["skank", "rest"], fold_into="rest"``
    for a two-stem specialist trained on drums/bass/rest backgrounds.
    """
    s = settings or MixSettings()
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    missing = [layer.slot for layer in layers if layer.slot not in slots]
    if missing:
        raise ValueError(f"layers target slots {missing} that are not in {list(slots)}")
    backgrounds = sorted(p for d in map(Path, background_dirs) for p in d.iterdir() if p.is_dir())
    pools = [sorted(p for p in Path(layer.dir).iterdir() if p.suffix in (".flac", ".wav")) for layer in layers]
    if not backgrounds or not all(pools):
        raise ValueError(f"need backgrounds ({len(backgrounds)}) and examples for every layer "
                         f"({ {layer.name: len(pool) for layer, pool in zip(layers, pools)} })")
    n = int(round(s.chunk_s * SR))
    # Slots only layers write to (e.g. acid) are not read from the backgrounds.
    new_slots = {layer.slot for layer in layers}
    bg_slots = [x for x in slots if x not in new_slots or _has_slot(backgrounds[0], x)]
    folded = []
    if fold_into:
        if fold_into not in slots:
            raise ValueError(f"fold_into {fold_into!r} is not in {list(slots)}")
        folded = sorted({f.stem for f in backgrounds[0].iterdir() if f.suffix in (".flac", ".wav")}
                        - set(slots) - {"mixture"})
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
            if len(cache) >= cache_size:  # keep memory bounded on large datasets
                cache.pop(next(iter(cache)))
            loaded = _load_slots(bg_path, bg_slots + folded)
            for k in folded:
                loaded[fold_into] = loaded.get(fold_into, 0) + loaded.pop(k)
            cache[bg_path] = loaded
        bg = cache[bg_path]
        length = len(next(iter(bg.values())))
        if length < n:
            continue
        off = int(rng.integers(length - n + 1))
        chunk = {k: v[off : off + n] for k, v in bg.items()}
        bg_rms = _rms(sum(chunk.values()))
        if bg_rms < s.min_background_rms:
            continue
        if s.min_slot_rms and min(_rms(v) for v in chunk.values()) < s.min_slot_rms:
            continue

        stems = {k: chunk.get(k, np.zeros((n, 2), np.float32)) for k in slots}
        meta = {"background": str(bg_path), "offset": off, "layers": {}}
        for layer, pool in zip(layers, pools):
            meta["layers"][layer.name] = None
            if rng.random() >= layer.prob:
                continue
            path = pool[rng.integers(len(pool))]
            line, _ = sf.read(path, dtype="float32", always_2d=True)
            if len(line) < n:  # loop short examples to fill the chunk
                line = np.tile(line, (int(np.ceil(n / len(line))), 1))
            l_off = int(rng.integers(len(line) - n + 1))
            audio = line[l_off : l_off + n]
            rel_db = rng.uniform(*layer.rel_db)
            stems[layer.slot] = stems[layer.slot] + audio * (bg_rms * 10 ** (rel_db / 20) / _rms(audio))
            meta["layers"][layer.name] = {"file": str(path), "offset": l_off, "rel_db": rel_db}

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


def _has_slot(folder: Path, name: str) -> bool:
    return any((folder / f"{name}{ext}").exists() for ext in (".flac", ".wav"))


def acid_mix(background_dirs: list[str | Path], acid_dir: str | Path, out_dir: str | Path, count: int,
             slots: list[str] = ("drums", "bass", "acid", "rest"), acid_slot: str = "acid",
             seed: int = 0, settings: MixSettings | None = None, start_index: int = 0,
             with_mixture: bool = False, fmt: str = "flac", cache_size: int = 64) -> list[Path]:
    """``synth_mix`` with a single acid layer (the acid-v1 training set)."""
    s = settings or MixSettings()
    layer = Layer(slot=acid_slot, dir=acid_dir, prob=s.acid_prob, rel_db=s.acid_rel_db, name="acid")
    out = synth_mix(background_dirs, [layer], out_dir, count, list(slots), seed=seed, settings=s,
                    start_index=start_index, with_mixture=with_mixture, fmt=fmt, cache_size=cache_size)
    for dest in out:  # keep the meta layout acid-v1 datasets were written with
        meta = json.loads((dest / "meta.json").read_text())
        meta["acid"] = meta.pop("layers")["acid"]
        (dest / "meta.json").write_text(json.dumps(meta))
    return out


def _fold_one(args: tuple[Path, Path, list[str], str]) -> Path:
    src, dest, keep, into = args
    dest.mkdir(parents=True, exist_ok=True)
    total = None
    fmt = "flac"
    for f in sorted(src.iterdir()):
        name, ext = f.stem, f.suffix.lstrip(".")
        if ext not in FORMATS:
            if f.is_file():  # meta.json and the like
                (dest / f.name).write_bytes(f.read_bytes())
            continue
        if name in keep or name == "mixture":
            target = dest / f.name
            if not target.exists():
                os.link(f, target)  # same audio, no extra disk
            continue
        fmt = ext
        audio, _ = sf.read(f, dtype="float32", always_2d=True)
        total = audio if total is None else total + audio
    if total is not None:
        _write(dest / into, total, fmt)
    return dest


def fold_slots(src_dirs: list[str | Path], out_dir: str | Path, keep: list[str], into: str = "rest",
               workers: int = 1) -> list[Path]:
    """Copy example folders keeping the ``keep`` stems and summing every other stem into ``into``.

    Turns a 4-slot set (drums / bass / acid / rest) into the 2-slot set of an adapter
    specialist (acid / rest) without remixing, so both are trained on the same audio.
    Kept stems and mixtures are hard links.
    """
    from concurrent.futures import ProcessPoolExecutor

    out_dir = Path(out_dir)
    jobs = [(d, out_dir / d.name, list(keep), into)
            for s in map(Path, src_dirs) for d in sorted(s.iterdir()) if d.is_dir()]
    with ProcessPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(_fold_one, jobs, chunksize=8))
