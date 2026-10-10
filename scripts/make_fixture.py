"""Regenerate fixtures/contract/: a shared test case for priism and MyMixxx.

A 10 s synthetic track (no copyrighted audio), its separation with the ``fake`` profile,
a project.json and the exported .stem.mp4. MyMixxx parses the manifest and project and
loads the stem file in Mixxx; priism's tests check the files still match the schemas.

    python scripts/make_fixture.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine_helpers import synth_track  # noqa: E402
from priism.engine.export_ni import export_ni  # noqa: E402
from priism.engine.profile import load_profile  # noqa: E402
from priism.engine.separate import separate  # noqa: E402

OUT = ROOT / "fixtures" / "contract"


def pcm_sha256(path: Path) -> str:
    """MyMixxx's track identity: SHA-256 of the decoded audio, interleaved float32 little-endian."""
    x, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return hashlib.sha256(x.astype("<f4").tobytes()).hexdigest()


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    src = OUT / "source.wav"
    sf.write(str(src), synth_track(10.0, 44100, seed=1), 44100, subtype="PCM_16")
    separate(src, OUT, load_profile("fake"), source_path="source.wav", source_sha256=pcm_sha256(src),
             source_decoder="priism:fixture")
    (OUT / "project.json").write_text(json.dumps({
        "schema_version": "1.1",
        "manifest": "manifest.json",
        "slots": [
            {"name": "Highs", "color": "#FF5A3C", "stems": ["high"]},
            {"name": "Lows", "color": "#3C8CFF", "stems": ["low"], "gain": [1.0]},
            {"name": "Mids", "color": "#FFD23C", "stems": ["mid"]},
            {"name": "Rest", "color": "#3CFF8C", "stems": ["residual"]},
        ],
        "discarded": [],
    }, indent=2) + "\n", encoding="utf-8")
    report = export_ni(OUT / "project.json", OUT / "fixture.stem.mp4")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
