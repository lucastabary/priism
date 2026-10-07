"""Split tracks into the slots of a profile with a pretrained model.

A profile names a model and says how its raw stems fill each output slot,
so the same code serves dub/acid today and any other genre later. One slot
may be ``"residual"``: it gets the mix minus every other slot, which keeps
the slots summing back exactly to the original track.
"""

from __future__ import annotations

import json
import re
import tempfile
import tomllib
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}
RESIDUAL = "residual"


@dataclass
class Profile:
    name: str
    model: str
    slots: dict[str, list[str] | str]  # slot name -> model stems, or "residual"
    sample_rate: int = 44100

    def __post_init__(self) -> None:
        residuals = [k for k, v in self.slots.items() if v == RESIDUAL]
        if len(residuals) > 1:
            raise ValueError(f"profile {self.name}: only one slot can be residual, got {residuals}")
        for k, v in self.slots.items():
            if v != RESIDUAL and (not isinstance(v, list) or not v):
                raise ValueError(f"profile {self.name}: slot {k!r} needs a list of model stems or \"residual\"")


def load_profile(name_or_path: str | Path) -> Profile:
    path = Path(name_or_path)
    if not path.exists():
        path = PROFILES_DIR / f"{name_or_path}.toml"
    data = tomllib.loads(path.read_text())
    slots = {k: ([s.lower() for s in v] if isinstance(v, list) else v) for k, v in data["slots"].items()}
    return Profile(name=data.get("name", path.stem), model=data["model"], slots=slots,
                   sample_rate=data.get("sample_rate", 44100))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad an (n, channels) array along time."""
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x), x.shape[1]), dtype=x.dtype)])


def assemble_slots(mix: np.ndarray, stems: dict[str, np.ndarray], profile: Profile) -> dict[str, np.ndarray]:
    """Sum model stems into profile slots; the residual slot takes what is left of the mix."""
    n = len(mix)
    out: dict[str, np.ndarray] = {}
    for slot, sources in profile.slots.items():
        if sources == RESIDUAL:
            continue
        missing = [s for s in sources if s not in stems]
        if missing:
            raise KeyError(f"model {profile.model} has no stem {missing} for slot {slot!r}; it gives {sorted(stems)}")
        out[slot] = sum(_fit(stems[s], n) for s in sources)
    for slot, sources in profile.slots.items():
        if sources == RESIDUAL:
            out[slot] = mix - sum(out.values()) if out else mix.copy()
    return {slot: out[slot].astype(np.float32) for slot in profile.slots}


def read_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Read as float32 (n, 2) at ``sample_rate``."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if sr != sample_rate:
        g = gcd(sr, sample_rate)
        x = resample_poly(x, sample_rate // g, sr // g, axis=0).astype(np.float32)
    return x


_STEM_RE = re.compile(r"\(([^)]+)\)")


def run_model(track: Path, model: str, sample_rate: int, model_dir: str | None = None) -> dict[str, np.ndarray]:
    """Raw stems from python-audio-separator, keyed by lower-case stem name."""
    try:
        from audio_separator.separator import Separator
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit("python-audio-separator is missing: pip install 'priism[separate]' (or 'priism[separate-cpu]' without an NVIDIA GPU)") from e

    with tempfile.TemporaryDirectory() as tmp:
        # The default 0.9 threshold rescales loud stems one by one, which breaks
        # the sum back to the mix; 1.0 only rescales what would clip anyway.
        kwargs = {"output_dir": tmp, "output_format": "WAV", "sample_rate": sample_rate,
                  "normalization_threshold": 1.0}
        if model_dir:
            kwargs["model_file_dir"] = model_dir
        sep = Separator(**kwargs)
        sep.load_model(model_filename=model)
        stems = {}
        for f in sep.separate(str(track)):
            path = Path(f) if Path(f).is_absolute() else Path(tmp) / f
            # Output files are named like "<track>_(Drums)_<model>.wav".
            names = _STEM_RE.findall(path.stem)
            if names:
                stems[names[-1].lower()] = read_audio(path, sample_rate)
        return stems


def separate_track(track: str | Path, profile: Profile, out_dir: str | Path,
                   model_dir: str | None = None, overwrite: bool = False) -> Path:
    """Write one WAV per slot plus a manifest under ``out_dir/<track name>/``."""
    track = Path(track)
    dest = Path(out_dir) / track.stem
    manifest = dest / "manifest.json"
    if manifest.exists() and not overwrite:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    mix = read_audio(track, profile.sample_rate)
    stems = run_model(track, profile.model, profile.sample_rate, model_dir)
    slots = assemble_slots(mix, stems, profile)
    for slot, audio in slots.items():
        sf.write(dest / f"{slot}.wav", audio, profile.sample_rate, subtype="FLOAT")
    manifest.write_text(json.dumps({
        "track": str(track),
        "profile": profile.name,
        "model": profile.model,
        "slots": {k: v for k, v in profile.slots.items()},
        "model_stems": sorted(stems),
    }, indent=1))
    return dest


def find_tracks(inputs: list[str | Path]) -> list[Path]:
    tracks: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            tracks += sorted(p for p in item.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        else:
            tracks.append(item)
    return tracks
