import numpy as np
import soundfile as sf

from priism.lab.gen import fxchain


def test_every_effect_renders(tmp_path, monkeypatch):
    sr, bpm = 22050, 124.0
    ir = np.random.default_rng(0).normal(0, 1, (sr // 2, 2)) * np.exp(-np.arange(sr // 2) / 2000)[:, None]
    sf.write(tmp_path / "room.wav", ir, sr)
    monkeypatch.setattr(fxchain, "IR_DIR", str(tmp_path))
    fxchain._IRS.clear()
    rng = np.random.default_rng(1)
    dry = np.random.default_rng(2).normal(0, 0.2, 2 * sr)
    types = sorted({t for _, ts in fxchain.CATEGORIES.values() for t in ts})
    for t in types:
        node = fxchain._node(t, "lead", rng)
        node["category"] = "x"
        if t in fxchain.AUTOMATABLE:
            fxchain._automate([node], rng, 8)
        y = fxchain.apply_chain(dry, [{"order": "usual"}, node], sr, bpm)
        assert y.shape == (len(dry), 2) and np.all(np.isfinite(y)), t
        d = fxchain.apply_chain(dry, [{"order": "usual"}, node], sr, bpm, space=False)
        if t in fxchain.SPACE and "auto" not in node:  # an automated send may sit at 0 for these 2 s
            assert not np.allclose(y, d), t  # the send adds a wet part
    fxchain._IRS.clear()


def test_chains_usual_order_and_curves():
    rng = np.random.default_rng(3)
    usual = 0
    for _ in range(200):
        chain = fxchain.sample_chain("pad", rng, 32)
        ranks = [fxchain.CATEGORIES[n["category"]][0] for n in chain[1:]]
        if chain[0]["order"] == "usual":
            usual += 1
            assert ranks == sorted(ranks)
        assert sum(n["type"] == "pan" for n in chain[1:]) == 1
    assert 120 < usual < 190  # ~80 % usual orders
    t = np.linspace(0, 60, 5000)
    for shape in ("lfo", "ramp", "steps"):
        a = {"shape": shape, "period_bars": 4.0, "phase": 0.3, "wave": "triangle", "up": True, "seed": 5}
        c = fxchain.curve(a, t, 120.0)
        assert c.min() >= -1 - 1e-9 and c.max() <= 1 + 1e-9 and c.std() > 0.1


def test_deep_chains_in_songs(monkeypatch):
    import priism.lab.gen.song as song

    base_mix, _, _ = song.render_song(7, duration_s=8, sample_rate=22050)
    monkeypatch.setattr(song, "FX_CHAIN_P", 1.0)
    mix, tracks, meta = song.render_song(7, duration_s=8, sample_rate=22050)
    np.testing.assert_allclose(mix, np.sum(tracks, axis=0), atol=1e-5)
    assert np.all(np.isfinite(mix)) and not np.allclose(mix, base_mix)
    for s in meta["sources"]:
        secs = s["sections"]
        assert secs[0]["start_s"] == 0 and all(a["end_s"] == b["start_s"] for a, b in zip(secs, secs[1:]))
        assert abs(secs[-1]["end_s"] - meta["duration_s"]) < 1e-3
        assert {"pan", "width", "wet_ratio_db", "rt60_s", "centroid_hz", "thd", "chain_id"} <= set(secs[0])
        assert s["chain"] and len(s["fx_chains"]) >= 1
    monkeypatch.setattr(song, "FX_CHAIN_P", 0.0)
    again, _, _ = song.render_song(7, duration_s=8, sample_rate=22050)
    np.testing.assert_array_equal(again, base_mix)  # off: songs as before
