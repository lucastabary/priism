import json

import numpy as np
import pytest
import soundfile as sf

from priism import separate
from priism.separate import Profile, assemble_slots, load_profile


def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal((n, 2)).astype(np.float32) * 0.1


def test_bundled_profiles_load():
    for name in ("standard", "dub-acid-baseline"):
        p = load_profile(name)
        assert len(p.slots) == 4
        assert list(p.slots.values()).count("residual") == 1


def test_slots_sum_back_to_mix():
    stems = {k: _noise(1000, i) for i, k in enumerate(["drums", "bass", "other", "vocals", "piano"])}
    mix = sum(stems.values()) + _noise(1000, 99) * 0.01  # a model never sums exactly
    profile = Profile("t", "m", {"drums": ["drums"], "bass": ["bass"], "acid": ["other"], "rest": "residual"})
    slots = assemble_slots(mix, stems, profile)
    assert list(slots) == ["drums", "bass", "acid", "rest"]
    np.testing.assert_allclose(sum(slots.values()), mix, atol=1e-5)
    np.testing.assert_allclose(slots["rest"], stems["vocals"] + stems["piano"] + mix - sum(stems.values()), atol=1e-5)


def test_slots_can_merge_stems_and_fit_length():
    stems = {"a": _noise(900, 1), "b": _noise(1100, 2)}
    profile = Profile("t", "m", {"ab": ["a", "b"]})
    out = assemble_slots(_noise(1000, 3), stems, profile)
    assert out["ab"].shape == (1000, 2)


def test_missing_stem_is_a_clear_error():
    with pytest.raises(KeyError, match="acid"):
        assemble_slots(_noise(10, 0), {"drums": _noise(10, 1)}, Profile("t", "m", {"acid": ["acid"]}))


def test_profile_rejects_two_residuals():
    with pytest.raises(ValueError):
        Profile("t", "m", {"a": "residual", "b": "residual"})


def test_separate_track_writes_slots_and_skips_done(tmp_path, monkeypatch):
    mix = _noise(4410, 0)
    track = tmp_path / "song.wav"
    sf.write(track, mix, 44100, subtype="FLOAT")
    calls = []

    def fake_model(path, model, sr, model_dir=None):
        calls.append(path)
        return {"drums": mix * 0.5, "bass": mix * 0.25, "other": mix * 0.125}

    monkeypatch.setattr(separate, "run_model", fake_model)
    profile = load_profile("dub-acid-baseline")
    dest = separate.separate_track(track, profile, tmp_path / "out")
    for slot in ("drums", "bass", "acid", "rest"):
        assert (dest / f"{slot}.wav").exists()
    rest, _ = sf.read(dest / "rest.wav")
    np.testing.assert_allclose(rest, mix * 0.125, atol=1e-6)
    assert json.loads((dest / "manifest.json").read_text())["profile"] == "dub-acid-baseline"
    separate.separate_track(track, profile, tmp_path / "out")
    assert len(calls) == 1
