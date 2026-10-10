import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from engine_helpers import synth_track, validate
from priism.cli import main
from priism.engine.manifest import Manifest
from priism.engine.mp4 import read_stem_box
from priism.engine.profile import load_profile
from priism.engine.separate import separate

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")


def _load_all(out):
    m = Manifest.from_dict(json.loads((out / "manifest.json").read_text()))
    stems = {s.id: sf.read(out / s.file, dtype="float64", always_2d=True) for s in m.stems}
    stems["residual"] = sf.read(out / m.residual_file, dtype="float64", always_2d=True)
    return m, stems


@pytest.mark.parametrize("sr, subtype, mono", [(44100, "PCM_16", False), (48000, "FLOAT", False),
                                               (22050, "PCM_24", True)])
def test_fake_separation_keeps_rate_frames_and_sums_to_the_source(tmp_path, sr, subtype, mono):
    x = synth_track(2.5, sr)
    if mono:
        x = x[:, :1]
    sf.write(tmp_path / "in.wav", x, sr, subtype=subtype)
    src, _ = sf.read(tmp_path / "in.wav", dtype="float64", always_2d=True)
    out = tmp_path / "out"
    separate(tmp_path / "in.wav", out, load_profile("fake"), source_decoder="test")

    raw = json.loads((out / "manifest.json").read_text())
    validate(raw, "manifest.schema.json")
    m, stems = _load_all(out)
    assert m.source.sample_rate == sr and m.source.frames == len(src) and m.source.channels == (1 if mono else 2)
    assert m.source.decoder == "test"
    assert [s.id for s in m.stems] == ["low", "mid", "high"]
    assert [s.parent for s in m.stems] == ["mix", "upper", "upper"]
    for audio, rate in stems.values():
        assert rate == sr and audio.shape == (len(src), 2)
    if mono:
        src = np.repeat(src, 2, axis=1)
    total = sum(a for a, _ in stems.values())
    err = np.max(np.abs(total - src))
    # Exact for integer PCM sources; float sources only lose the residual's 24-bit rounding.
    assert err == 0.0 if subtype != "FLOAT" else err <= 2.0**-24 + 1e-12
    # Each band holds real signal (not everything in the residual).
    for s in m.stems:
        assert np.std(stems[s.id][0]) > 1e-3


def test_a_loud_stem_goes_to_float_wav_instead_of_clipping(tmp_path, monkeypatch):
    from priism.engine.backends import fake

    real = fake.FakeBands.separate

    def louder(self, audio, progress):
        out = real(self, audio, progress)
        out["low"] = out["low"] * 20  # far above full scale; the residual compensates
        return out

    monkeypatch.setattr(fake.FakeBands, "separate", louder)
    sf.write(tmp_path / "in.wav", synth_track(1.0), 44100, subtype="PCM_16")
    separate(tmp_path / "in.wav", tmp_path / "out", load_profile("fake"))
    m, stems = _load_all(tmp_path / "out")
    assert m.stems[0].file.endswith(".wav") and np.max(np.abs(stems["low"][0])) > 1
    src, _ = sf.read(tmp_path / "in.wav", dtype="float64")
    # The residual compensating it is above full scale too, so float32: relative precision 2**-24.
    bound = 2.0**-24 * np.max(np.abs(stems["residual"][0]))
    assert np.max(np.abs(sum(a for a, _ in stems.values()) - src)) <= bound


def test_a_failed_run_leaves_no_manifest(tmp_path, monkeypatch):
    from priism.engine.backends import fake

    sf.write(tmp_path / "in.wav", synth_track(1.0), 44100)
    out = tmp_path / "out"
    separate(tmp_path / "in.wav", out, load_profile("fake"))
    assert (out / "manifest.json").exists()

    def boom(self, audio, progress):
        raise RuntimeError("killed")

    monkeypatch.setattr(fake.FakeBands, "separate", boom)
    with pytest.raises(RuntimeError):
        separate(tmp_path / "in.wav", out, load_profile("fake"))
    assert not (out / "manifest.json").exists()


def _project(out, slots):
    colors = ["#FF5A3C", "#3C8CFF", "#FFD23C", "#3CFF8C"]
    (out / "project.json").write_text(json.dumps({
        "schema_version": "1.1", "manifest": "manifest.json", "discarded": [],
        "slots": [{"name": f"S{i}", "color": colors[i], "stems": s} for i, s in enumerate(slots)]}))


def _decode(path, stream):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{stream}", "-f", "f32le",
                          "-ac", "2", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["alac", "aac"])
def test_export_ni_is_aligned_and_carries_the_stem_box(tmp_path, capsys, codec):
    sf.write(tmp_path / "in.wav", synth_track(3.0), 44100, subtype="PCM_16")
    out = tmp_path / "out"
    main(["separate", str(tmp_path / "in.wav"), "--out", str(out), "--profile", "fake"])
    _project(out, [["high"], ["low"], ["mid"], ["residual"]])
    capsys.readouterr()
    main(["export-ni", str(out), "--out", str(tmp_path / "x.stem.mp4"), "--codec", codec, "--progress", "jsonl"])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "done" and events[-1]["clipped"] == [] and events[-1]["codec"] == codec
    assert len(events[-1]["peaks_dbfs"]["slots"]) == 4

    meta = json.loads(read_stem_box((tmp_path / "x.stem.mp4").read_bytes()))
    assert meta["version"] == 1 and [s["name"] for s in meta["stems"]] == ["S0", "S1", "S2", "S3"]
    assert meta["mastering_dsp"]["limiter"]["enabled"] is False

    src, _ = sf.read(tmp_path / "in.wav", dtype="float64")
    _, stems = _load_all(out)
    refs = [src, stems["high"][0], stems["low"][0], stems["mid"][0], stems["residual"][0]]
    for k, ref in enumerate(refs):
        dec = _decode(tmp_path / "x.stem.mp4", k)
        assert len(dec) == len(src)  # no priming or padding left (edit list for AAC)
        if codec == "alac":
            # Lossless: the fixture's stems are on the 24-bit grid, so the decode is bit-exact.
            assert np.max(np.abs(dec - ref)) < 1e-7
            continue
        # Lossy: the decoded track matches its reference better than at a 1-sample shift.
        n = slice(4410, len(src) - 4410)
        err0 = np.sum((dec[n] - ref[n]) ** 2)
        for shift in (-1, 1):
            assert err0 < np.sum((dec[n] - np.roll(ref, shift, axis=0)[n]) ** 2)


@needs_ffmpeg
def test_export_ni_clipping_is_an_error_with_alac_and_a_warning_with_aac(tmp_path, capsys):
    sf.write(tmp_path / "in.wav", synth_track(1.0), 44100, subtype="PCM_16")
    out = tmp_path / "out"
    separate(tmp_path / "in.wav", out, load_profile("fake"))
    _project(out, [["high"], ["low"], ["mid"], ["residual"]])
    p = json.loads((out / "project.json").read_text())
    p["slots"][2]["gain"] = [8.0]
    (out / "project.json").write_text(json.dumps(p))
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        main(["export-ni", str(out), "--out", str(tmp_path / "y.stem.mp4"), "--progress", "jsonl"])
    assert e.value.code == 1
    err = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert err["event"] == "error" and err["clipped"] == ["S2"] and err["peaks_dbfs"]["slots"][2] > 0
    assert not (tmp_path / "y.stem.mp4").exists()
    for args in (["--codec", "aac"], ["--on-clip", "warn"]):
        main(["export-ni", str(out), "--out", str(tmp_path / "x.stem.mp4"), "--progress", "jsonl", *args])
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert any(e["event"] == "warning" for e in events) and events[-1]["clipped"] == ["S2"]


def test_cli_version_and_errors(tmp_path, capsys):
    main(["version", "--json"])
    info = json.loads(capsys.readouterr().out)
    assert info["schema_version"] == "1.2" and "fake" in info["profiles"] and "fake:bands" in info["models"]
    with pytest.raises(SystemExit) as e:
        main(["separate", str(tmp_path / "missing.wav"), "--out", str(tmp_path / "o"), "--profile", "fake",
              "--progress", "jsonl"])
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1])["event"] == "error" and "error:" in captured.err
