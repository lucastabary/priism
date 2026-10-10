"""``priism separate``: run a profile's separation tree on a track, write stems + manifest.

Output layout::

    <out>/manifest.json          written last, atomically (absent = not finished)
    <out>/stems/<id>.flac|.wav   one file per stem, plus residual.flac|.wav

Every stem has the input's sample rate and exact frame count; models running at another
rate are resampled in and out with a zero-phase filter, so nothing shifts in time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import __version__
from .audio import fit, read_track, resample, sha256_file, write_lossless, write_residual
from .backends import Model, load_model
from .manifest import RESIDUAL, ROOT, Manifest, Profile as ProfileRef, Source, Stem, write_json_atomic
from .profile import Profile
from .progress import Reporter

MANIFEST = "manifest.json"
STEMS_DIR = "stems"


def separate(input_path: str | Path, out_dir: str | Path, profile: Profile, *, source_path: str | None = None,
             source_sha256: str | None = None, source_decoder: str | None = None, device: str = "cpu",
             reporter: Reporter | None = None) -> Path:
    reporter = reporter or Reporter()
    input_path, out_dir = Path(input_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST
    # A manifest from an earlier run would describe stems this run is about to overwrite.
    manifest_path.unlink(missing_ok=True)
    (out_dir / STEMS_DIR).mkdir(exist_ok=True)

    reporter.progress("decode", 0.0)
    track = read_track(input_path)
    sr, frames = track.sample_rate, track.frames
    source = Source(
        path=source_path or str(input_path),
        sha256=source_sha256 or sha256_file(input_path),
        sample_rate=sr, channels=track.channels, frames=frames,
        decoder=source_decoder or track.decoder,
    )
    reporter.progress("decode", 1.0)

    models: dict[str, Model] = {}
    for spec in profile.models:
        reporter.progress(f"load:{spec}", 0.0)
        models[spec] = load_model(spec, device=device)
        reporter.progress(f"load:{spec}", 1.0)

    # Signals are float64 so the subtractions of the tree add no rounding of their own.
    mix = track.audio.astype(np.float64)
    signals: dict[str, np.ndarray] = {ROOT: mix}
    lineage: dict[str, list[str]] = {ROOT: []}  # models each signal went through
    parents: dict[str, str] = {}
    for i, step in enumerate(profile.steps):
        stage = f"step{i + 1}:{step.model}"
        model = models[step.model]
        x = signals[step.input]
        raw = model.separate(resample(x.astype(np.float32), sr, model.sample_rate), reporter.stage(stage))
        missing = [s for o in step.outputs for s in o.sources if s not in raw]
        if missing:
            raise RuntimeError(f"{step.model} gives {sorted(raw)}, not {missing}")
        back = {s: fit(resample(a, model.sample_rate, sr), frames).astype(np.float64) for s, a in raw.items()}
        named = 0.0
        for o in step.outputs:
            y = sum(back[s] for s in o.sources)
            signals[o.id], lineage[o.id], parents[o.id] = y, lineage[step.input] + [step.model], step.input
            named = named + y
        if step.rest:
            r = step.rest.id
            signals[r], lineage[r], parents[r] = x - named, lineage[step.input] + [step.model], step.input
        reporter.progress(stage, 1.0)

    # Write stems, then the residual from what the stem files hold, so the files sum to the source.
    reporter.progress("write", 0.0)
    finals = profile.final_outputs()
    stems: list[Stem] = []
    stored_sum = np.zeros_like(mix)
    for k, (_, o) in enumerate(finals):
        dest = out_dir / STEMS_DIR / o.id
        _clear(dest)
        path, stored = write_lossless(dest, signals[o.id].astype(np.float32), sr)
        stored_sum += stored
        stems.append(Stem(id=o.id, label=o.label, kind=o.kind, file=f"{STEMS_DIR}/{path.name}",
                          models=list(dict.fromkeys(lineage[o.id])), parent=parents[o.id]))
        reporter.progress("write", (k + 1) / (len(finals) + 1))
    dest = out_dir / STEMS_DIR / RESIDUAL
    _clear(dest)
    residual_path = write_residual(dest, mix - stored_sum, sr)
    reporter.progress("write", 1.0)

    manifest = Manifest(
        priism_version=__version__,
        profile=ProfileRef(name=profile.name, version=profile.version),
        source=source,
        models={spec: m.info() for spec, m in models.items()},
        stems=stems,
        residual_file=f"{STEMS_DIR}/{residual_path.name}",
        suggested_slots=profile.suggested_slots or suggest_slots(stems),
    )
    manifest.validate()
    write_json_atomic(manifest_path, manifest.to_dict())
    return manifest_path


def _clear(dest_no_ext: Path) -> None:
    for ext in (".flac", ".wav"):
        dest_no_ext.with_suffix(ext).unlink(missing_ok=True)


def suggest_slots(stems: list[Stem]) -> list[list[str]]:
    """Default NI grouping: drums, bass, everything else + residual, vocals (empty groups dropped)."""
    groups: dict[str, list[str]] = {"drums": [], "bass": [], "music": [], "vocals": []}
    for s in stems:
        groups[s.kind if s.kind in ("drums", "bass", "vocals") else "music"].append(s.id)
    groups["music"].append(RESIDUAL)
    return [g for g in groups.values() if g]
