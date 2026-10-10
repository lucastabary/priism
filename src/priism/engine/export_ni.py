"""``priism export-ni``: a Native Instruments ``.stem.mp4`` from a manifest and a project.

Layout (the NI Stem format, as Mixxx's ``SoundSourceSTEM`` and Traktor read it):

- 5 stereo AAC tracks, same rate: track 0 the full mix, tracks 1-4 the project's slots
  (slot audio = sum of gain x stem);
- a ``moov/udta/stem`` box holding JSON: slot names and colours, mastering DSP disabled
  (slots must sum to the mix as they are).

AAC adds an encoder delay (priming samples) in front of the audio; ffmpeg's MP4 muxer
records it in an edit list, which decoders that honour edit lists (FFmpeg, so Mixxx's
``SoundSourceSTEM``) skip, keeping frame 0 on frame 0 of the source.

Track 0 is rebuilt as the sum of every stem plus the residual, which is the decoded source
by construction, so the export does not need the original file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .manifest import RESIDUAL, ContractError, Manifest, Project, read_json
from .mp4 import add_stem_box
from .progress import Reporter

AAC_BITRATE = "256k"

# NI's reference values, all disabled: Traktor would otherwise compress and limit the slots.
MASTERING_DSP = {
    "compressor": {"enabled": False, "input_gain": 0.5, "output_gain": 0.5, "threshold": 0.0, "dry_wet": 100,
                   "attack": 0.003, "release": 0.3, "ratio": 16, "hp_cutoff": 300},
    "limiter": {"enabled": False, "release": 0.05, "threshold": 0.0, "ceiling": -0.35},
}


def stem_metadata(project: Project) -> dict:
    return {"mastering_dsp": MASTERING_DSP, "version": 1,
            "stems": [{"name": s.name, "color": s.color.upper()} for s in project.slots]}


def load_inputs(project_path: Path) -> tuple[Project, Manifest, Path]:
    project = Project.from_dict(read_json(project_path))
    manifest_path = project_path.parent / project.manifest
    if not manifest_path.exists():
        raise ContractError(f"{project_path}: manifest {project.manifest!r} not found")
    manifest = Manifest.from_dict(read_json(manifest_path))
    project.validate(manifest)
    return project, manifest, manifest_path.parent


def _read(root: Path, rel: str, manifest: Manifest) -> np.ndarray:
    x, sr = sf.read(str(root / rel), dtype="float64", always_2d=True)
    if sr != manifest.source.sample_rate or len(x) != manifest.source.frames or x.shape[1] != 2:
        raise ContractError(f"{rel}: {sr} Hz, {len(x)} frames, {x.shape[1]} ch; the manifest says "
                            f"{manifest.source.sample_rate} Hz, {manifest.source.frames} frames, stereo")
    return x


def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x), initial=0.0))
    return round(20 * np.log10(p), 2) if p > 0 else -120.0


def export_ni(project_path: str | Path, out_path: str | Path, *, fail_on_clip: bool = False,
              bitrate: str = AAC_BITRATE, reporter: Reporter | None = None) -> dict:
    reporter = reporter or Reporter()
    project_path, out_path = Path(project_path), Path(out_path)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("export-ni needs ffmpeg on PATH")
    project, manifest, root = load_inputs(project_path)
    files = {s.id: s.file for s in manifest.stems} | {RESIDUAL: manifest.residual_file}

    reporter.progress("mixdown", 0.0)
    cache: dict[str, np.ndarray] = {}

    def stem(i: str) -> np.ndarray:
        if i not in cache:
            cache[i] = _read(root, files[i], manifest)
        return cache[i]

    slots = []
    for k, slot in enumerate(project.slots):
        y = np.zeros((manifest.source.frames, 2))
        for i, g in zip(slot.stems, slot.gains()):
            y += g * stem(i)
        slots.append(y)
        reporter.progress("mixdown", (k + 1) / 5)
    mix = sum(stem(i) for i in files)  # every stem + residual = the decoded source
    reporter.progress("mixdown", 1.0)

    # Track order of the file: the mix, then the slots. Clipped tracks are named ("mix" or the slot name).
    names = ["mix"] + [s.name for s in project.slots]
    track_peaks = [peak_dbfs(y) for y in [mix] + slots]
    peaks = {"mix": track_peaks[0], "slots": track_peaks[1:]}
    clipped = [n for n, p in zip(names, track_peaks) if p > 0.0]
    if clipped:
        msg = f"peaks above 0 dBFS in {clipped} " \
              f"({', '.join(f'{p:+.2f} dB' for p in track_peaks if p > 0.0)})"
        if fail_on_clip:
            raise ContractError(msg)
        reporter.warning(msg + "; kept as is (AAC holds them, lower the gains to avoid clipping on output)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent, prefix=".priism-export-") as tmp:
        tmp = Path(tmp)
        inputs = []
        for k, y in enumerate([mix] + slots):
            p = tmp / f"track{k}.wav"
            sf.write(str(p), y.astype(np.float32), manifest.source.sample_rate, subtype="FLOAT")
            inputs += ["-i", str(p)]
        muxed = tmp / "muxed.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", *inputs]
        for k in range(5):
            cmd += ["-map", f"{k}:a:0"]
        cmd += ["-c:a", "aac", "-b:a", bitrate, "-map_metadata", "-1", "-fflags", "+bitexact",
                "-disposition:a:0", "default"] + [x for k in range(1, 5) for x in (f"-disposition:a:{k}", "0")]
        cmd += ["-f", "mp4", str(muxed)]
        reporter.progress("encode", 0.0)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()[-2000:]}")
        reporter.progress("encode", 1.0)
        payload = json.dumps(stem_metadata(project), separators=(",", ":")).encode("utf-8")
        final = tmp / "final.mp4"
        final.write_bytes(add_stem_box(muxed.read_bytes(), payload))
        os.replace(final, out_path)  # same volume (tmp is next to the output), so atomic
    return {"output": str(out_path), "peaks_dbfs": peaks, "clipped": clipped}
