import json

import numpy as np
import soundfile as sf

from priism.dataset import MixSettings, acid_mix, restem

SR = 44100


def _tone(n, f, amp=0.1):
    t = np.arange(n) / SR
    return np.stack([amp * np.sin(2 * np.pi * f * t)] * 2, axis=1).astype(np.float32)


def _musdb_like(root, name, n):
    d = root / name
    d.mkdir(parents=True)
    stems = {"drums": _tone(n, 100), "bass": _tone(n, 50), "other": _tone(n, 440), "vocals": _tone(n, 880)}
    for k, v in stems.items():
        sf.write(d / f"{k}.wav", v, SR)
    sf.write(d / "mixture.wav", sum(stems.values()), SR)
    return stems


def test_restem_maps_stems_and_puts_the_rest_in_residual(tmp_path):
    stems = _musdb_like(tmp_path / "musdb", "song", SR)
    (tmp_path / "musdb" / "not_a_track").mkdir()
    out = restem([tmp_path / "musdb"], tmp_path / "slots", {"drums": ["drums"], "bass": ["bass"], "rest": "residual"})
    assert [p.name for p in out] == ["musdb__song"]
    rest, _ = sf.read(out[0] / "rest.flac")
    np.testing.assert_allclose(rest, stems["other"] + stems["vocals"], atol=1e-4)


def test_acid_mix_writes_exact_targets(tmp_path):
    n = 3 * SR
    _musdb_like(tmp_path / "musdb", "song", n)
    slots_dir = tmp_path / "slots"
    restem([tmp_path / "musdb"], slots_dir, {"drums": ["drums"], "bass": ["bass"], "rest": "residual"})
    acid_dir = tmp_path / "acid"
    acid_dir.mkdir()
    sf.write(acid_dir / "acid_0000000.wav", _tone(SR, 220, 0.3), SR, subtype="FLOAT")

    settings = MixSettings(chunk_s=2.0, acid_prob=0.5)
    out = acid_mix([slots_dir], acid_dir, tmp_path / "train", count=12, seed=1, settings=settings)
    assert len(out) == 12
    with_acid = 0
    for d in out:
        meta = json.loads((d / "meta.json").read_text())
        stems = {k: sf.read(d / f"{k}.flac")[0] for k in ("drums", "bass", "acid", "rest")}
        assert stems["acid"].shape == (2 * SR, 2)
        assert np.max(np.abs(sum(stems.values()))) <= 1.0
        if meta["acid"]:
            with_acid += 1
            assert np.abs(stems["acid"]).max() > 0
        else:
            assert np.abs(stems["acid"]).max() == 0
    assert 0 < with_acid < 12
