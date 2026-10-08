import numpy as np
import pytest

from priism.gen import song, surge
from priism.gen.tonal import Note

pytestmark = pytest.mark.skipif(not surge.available(), reason="surgepy not installed")


def test_notes_played_by_a_patch_of_each_kind():
    rng = np.random.default_rng(0)
    notes = [Note(start=4 * i, length=2, pitch=48 + 3 * i, vel=0.8) for i in range(8)]
    n = 44100 * 4
    for kind in surge.KINDS:
        y = surge.render_notes(notes, surge.sample_patch(kind, rng), n, 44100, 128.0)
        assert y.shape == (n,) and np.isfinite(y).all() and np.sqrt(np.mean(y ** 2)) > 1e-5


def test_song_with_surge_parts_stays_additive(monkeypatch, tmp_path):
    monkeypatch.setattr(song, "SURGE_P", 1.0)
    for seed in range(40):  # find a song with a part Surge can play
        plan = song.plan_song(seed, 10.0, None, 6)
        if any(s["kind"] in surge.KINDS and s["twin_of"] is None for s in plan["sources"]):
            break
    mix, tracks, meta = song.render_song(seed, 10.0, 44100, n_sources=6)
    assert any("surge_patch" in s["params"] for s in meta["sources"])
    np.testing.assert_allclose(np.sum(tracks, axis=0), mix, atol=1e-5)
