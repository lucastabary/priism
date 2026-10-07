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


def test_synth_mix_layers_new_slots_and_distractors(tmp_path):
    from priism.dataset import Layer, parse_layer, synth_mix

    n = 3 * SR
    _musdb_like(tmp_path / "musdb", "song", n)
    slots_dir = tmp_path / "slots"
    restem([tmp_path / "musdb"], slots_dir, {"drums": ["drums"], "bass": ["bass"], "rest": "residual"})
    for name, f in (("skank", 330), ("acid", 220)):
        (tmp_path / name).mkdir()
        sf.write(tmp_path / name / f"{name}_0000000.wav", _tone(SR, f, 0.3), SR, subtype="FLOAT")

    layers = [Layer("skank", tmp_path / "skank", prob=1.0), parse_layer(f"rest={tmp_path / 'acid'}:1.0:-6:-6")]
    layers[1].name = "acid_distractor"
    out = synth_mix([slots_dir], layers, tmp_path / "train", count=4, slots=["drums", "bass", "skank", "rest"],
                    seed=2, settings=MixSettings(chunk_s=2.0), with_mixture=True)
    bg_rest = sf.read(slots_dir / "musdb__song" / "rest.flac")[0]
    for d in out:
        meta = json.loads((d / "meta.json").read_text())
        assert set(meta["layers"]) == {"skank", "acid_distractor"}
        stems = {k: sf.read(d / f"{k}.flac")[0] for k in ("drums", "bass", "skank", "rest", "mixture")}
        np.testing.assert_allclose(stems["mixture"], sum(v for k, v in stems.items() if k != "mixture"), atol=1e-4)
        assert np.abs(stems["skank"]).max() > 0
        # The distractor went into rest on top of the background's own rest.
        assert np.std(stems["rest"]) > np.std(bg_rest) * meta["gain"] * 1.05


def test_parse_layer():
    from priism.dataset import parse_layer

    layer = parse_layer("rest=data/acid:0.3:-20:-6")
    assert (layer.slot, str(layer.dir), layer.prob, layer.rel_db) == ("rest", "data/acid", 0.3, (-20.0, -6.0))
    assert parse_layer("skank=x").prob == 0.8


def test_synth_mix_folds_background_slots_for_a_two_stem_specialist(tmp_path):
    from priism.dataset import Layer, synth_mix

    _musdb_like(tmp_path / "musdb", "song", 3 * SR)
    slots_dir = tmp_path / "slots"
    restem([tmp_path / "musdb"], slots_dir, {"drums": ["drums"], "bass": ["bass"], "rest": "residual"})
    (tmp_path / "skank").mkdir()
    sf.write(tmp_path / "skank" / "skank_0000000.wav", _tone(SR, 330, 0.3), SR, subtype="FLOAT")
    out = synth_mix([slots_dir], [Layer("skank", tmp_path / "skank", prob=0.0)], tmp_path / "train", count=2,
                    slots=["skank", "rest"], seed=0, settings=MixSettings(chunk_s=2.0), fold_into="rest")
    meta = json.loads((out[0] / "meta.json").read_text())
    rest = sf.read(out[0] / "rest.flac")[0]
    full = sum(sf.read(slots_dir / "musdb__song" / f"{k}.flac")[0] for k in ("drums", "bass", "rest"))
    n = len(rest)
    expected = full[meta["offset"] : meta["offset"] + n] * meta["gain"]
    if not np.allclose(rest, expected, atol=1e-3):  # channel swap
        expected = expected[:, ::-1]
    np.testing.assert_allclose(rest, expected, atol=1e-3)
    assert sorted(p.name for p in out[0].iterdir()) == ["meta.json", "rest.flac", "skank.flac"]
