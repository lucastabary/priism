import json

import numpy as np
import soundfile as sf

from priism.lab.gen.genres import GENRES
from priism.lab.gen.song import MAX_SOURCES, MIN_SOURCES, plan_song, render_song, write_song


def test_mix_is_sum_of_tracks():
    mix, tracks, meta = render_song(3, duration_s=8, sample_rate=22050)
    assert mix.shape[1] == 2
    assert len(tracks) == len(meta["sources"])
    np.testing.assert_allclose(mix, np.sum(tracks, axis=0), atol=1e-5)
    assert np.max(np.abs(mix)) <= 1.0


def test_same_seed_same_song():
    a, _, ma = render_song(11, duration_s=6, sample_rate=22050)
    b, _, mb = render_song(11, duration_s=6, sample_rate=22050)
    np.testing.assert_array_equal(a, b)
    assert [s["kind"] for s in ma["sources"]] == [s["kind"] for s in mb["sources"]]


def test_every_track_is_heard():
    for seed in range(3):
        _, tracks, meta = render_song(seed, duration_s=10, sample_rate=22050)
        for s, t in zip(meta["sources"], tracks):
            assert np.max(np.abs(t)) > 1e-4, (seed, s["kind"])


def test_source_count_and_genres_covered():
    counts, genres = set(), set()
    for seed in range(200):
        plan = plan_song(seed, duration_s=30)
        n = len(plan["sources"])
        assert MIN_SOURCES <= n <= MAX_SOURCES
        counts.add(n)
        genres.add(plan["genre"])
    assert counts == set(range(MIN_SOURCES, MAX_SOURCES + 1))
    assert genres == set(GENRES)


def test_twin_acid_lines_have_two_roles():
    for seed in range(400):
        plan = plan_song(seed, duration_s=30, genre="acid")
        twins = [s for s in plan["sources"] if s["twin_of"] is not None and s["kind"] == "acid"]
        if twins:
            orig = plan["sources"][twins[0]["twin_of"]]
            assert {orig["role"], twins[0]["role"]} == {"rhythmic", "melodic"}
            return
    raise AssertionError("no acid twin in 400 acid songs")


def test_write_song(tmp_path):
    folder = write_song(5, tmp_path, duration_s=6, sample_rate=22050)
    meta = json.loads((folder / "meta.json").read_text())
    mix, sr = sf.read(folder / "mix.flac")
    assert sr == 22050
    total = sum(sf.read(folder / s["file"])[0] for s in meta["sources"])
    np.testing.assert_allclose(mix, total, atol=1e-4)  # 24-bit files


def test_ambiguous_pair_splits_one_part():
    for seed in range(3000):
        plan = plan_song(seed, duration_s=20)
        pair = [s for s in plan["sources"] if s["split_notes"]]
        if pair:
            break
    else:
        raise AssertionError("no ambiguous pair in 3000 songs")
    a, b = sorted(pair, key=lambda s: s["twin_of"] is not None)  # the original first, whatever the ids
    assert a["merge_group"] == b["merge_group"] == a["id"] and b["twin_of"] == a["id"]
    _, tracks, meta = render_song(seed, duration_s=20, sample_rate=22050)
    ta, tb = tracks[a["id"]], tracks[b["id"]]
    assert np.max(np.abs(ta)) > 1e-4 and np.max(np.abs(tb)) > 1e-4
    assert meta["sources"][a["id"]]["params"]["patch"]["osc"] == meta["sources"][b["id"]]["params"]["patch"]["osc"]


def test_lossy_keeps_length_and_alignment():
    from priism.lab.gen.augment import lossy

    sr = 22050
    t = np.arange(sr * 3) / sr
    x = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)], axis=1).astype(np.float32) * 0.5
    y = lossy(x, sr, "mp3", 128)
    assert y.shape == x.shape
    err = np.sqrt(np.mean((y[sr:2 * sr] - x[sr:2 * sr]) ** 2)) / np.sqrt(np.mean(x**2))
    assert err < 0.2


def test_multi_twins_give_three_or_more_parts_of_one_instrument(monkeypatch):
    from priism.lab.gen import song

    base = [song.plan_song(seed, 30.0) for seed in range(40)]
    monkeypatch.setattr(song, "TWIN_P", 1.0)
    monkeypatch.setattr(song, "TWIN_EXTRA_P", 1.0)
    groups = 0
    for seed in range(40):
        plan = song.plan_song(seed, 30.0)
        by_src: dict = {}
        for s in plan["sources"]:
            if s["twin_of"] is not None and s["merge_group"] is None:
                by_src.setdefault(s["twin_of"], []).append(s)
        for orig, ts in by_src.items():
            assert all(t["kind"] == plan["sources"][orig]["kind"] for t in ts)
            groups += len(ts) >= 2
    assert groups >= 10
    monkeypatch.setattr(song, "TWIN_P", 0.3)
    monkeypatch.setattr(song, "TWIN_EXTRA_P", 0.0)
    assert [song.plan_song(seed, 30.0) for seed in range(40)] == base  # off by default: songs unchanged


def test_twin_spread_moves_the_twin_sound_only(monkeypatch):
    from priism.lab.gen import song

    monkeypatch.setattr(song, "TWIN_P", 1.0)
    base = {"cutoff": 800.0, "res": 0.5, "osc": "saw", "drive": 0.0}
    assert song._drift(base, song._PATCH_DRIFT, 1) == base  # spread 0: exact copy
    monkeypatch.setattr(song, "TWIN_SPREAD", 1.0)
    moved = song._drift(base, song._PATCH_DRIFT, 1)
    assert moved["osc"] == "saw" and moved["cutoff"] != 800.0 and base["cutoff"] == 800.0
    assert 100 <= moved["cutoff"] <= 12000 and 0 <= moved["res"] <= 0.9
    for seed in range(30):
        _, _, meta = song.render_song(seed, duration_s=4, sample_rate=22050)
        tw = [s for s in meta["sources"] if s["twin_of"] is not None and s["merge_group"] is None
              and s["kind"] not in song.DRUM_KINDS]
        if tw:
            orig = meta["sources"][tw[0]["twin_of"]]
            key = "synth" if orig["kind"] == "acid" else "patch"
            if key in orig["params"] and "surge_patch" not in orig["params"]:
                assert tw[0]["params"][key] != orig["params"][key]
                return
    raise AssertionError("no tonal twin in 30 songs")


def test_pedalboard_fx_change_only_the_sound(monkeypatch):
    from priism.lab.gen import fx as fxmod

    off_mix, _, off = render_song(5, duration_s=8, sample_rate=22050)
    monkeypatch.setattr(fxmod, "PB_FX", True)
    a, tracks, on = render_song(5, duration_s=8, sample_rate=22050)
    b, _, _ = render_song(5, duration_s=8, sample_rate=22050)
    np.testing.assert_array_equal(a, b)  # still fully determined by the seed
    assert [s["kind"] for s in on["sources"]] == [s["kind"] for s in off["sources"]]
    assert [s["gain_db"] for s in on["sources"]] == [s["gain_db"] for s in off["sources"]]
    assert all("pb" in s["fx"] for s in on["sources"]) and not any("pb" in s["fx"] for s in off["sources"])
    assert not np.array_equal(a, off_mix)
    assert np.all(np.isfinite(a)) and np.max(np.abs(a)) <= 1.0
    for t in tracks:
        assert np.max(np.abs(t)) > 1e-4


def test_bus_fx_share_rooms_and_keep_the_mix_a_sum(monkeypatch):
    from priism.lab.gen import song as songmod

    _, _, off = render_song(9, duration_s=8, sample_rate=22050, genre="dub")
    monkeypatch.setattr(songmod, "BUS_FX", True)
    mix, tracks, on = render_song(9, duration_s=8, sample_rate=22050, genre="dub")
    again, _, _ = render_song(9, duration_s=8, sample_rate=22050, genre="dub")
    np.testing.assert_array_equal(mix, again)
    np.testing.assert_allclose(mix, np.sum(tracks, axis=0), atol=1e-5)
    assert [s["gain_db"] for s in on["sources"]] == [s["gain_db"] for s in off["sources"]]
    rooms = on["bus_fx"]["rooms"]
    for s in on["sources"]:
        if s["fx"]["reverb_send"]:
            assert s["fx"]["room"] in rooms and s["fx"]["reverb_rt60_s"] == s["fx"]["room"]["rt60_s"]
    assert np.all(np.isfinite(mix)) and np.max(np.abs(mix)) <= 1.0


def test_twin_place_spreads_the_parts_of_one_instrument(monkeypatch):
    from priism.lab.gen import song

    monkeypatch.setattr(song, "TWIN_P", 1.0)
    monkeypatch.setattr(song, "TWIN_EXTRA_P", 0.5)
    for seed in range(20):
        _, _, meta = song.render_song(seed, duration_s=4, sample_rate=22050)
        fam = [s for s in meta["sources"] if s["twin_of"] is not None]
        if fam:
            break
    off_mix, _, off = song.render_song(seed, duration_s=4, sample_rate=22050)
    monkeypatch.setattr(song, "TWIN_PLACE", 1.0)
    on_mix, tracks, on = song.render_song(seed, duration_s=4, sample_rate=22050)
    np.testing.assert_allclose(on_mix, np.sum(tracks, axis=0), atol=1e-4)  # still a sum of the tracks
    root = fam[0]["twin_of"]
    ids = [root] + [s["id"] for s in on["sources"] if s["twin_of"] == root]
    pans = sorted(on["sources"][i]["fx"]["pan"] for i in ids)
    assert 0.5 <= pans[-1] - pans[0] <= 0.9  # apart, but not hard left/right (the model hears stereo)
    rev = sorted(on["sources"][i]["fx"]["reverb_send"] for i in ids)
    assert rev[0] == 0.0 and rev[-1] > 0.1  # one dry, one deep
    outside = [s["id"] for s in on["sources"] if s["id"] not in ids and s["twin_of"] is None
               and not any(o["twin_of"] == s["id"] for o in on["sources"])]
    for i in outside:  # other sources keep their effects
        assert on["sources"][i]["fx"] == off["sources"][i]["fx"]


def test_build_up_enters_one_at_a_time(monkeypatch):
    import priism.lab.gen.song as song

    base = plan_song(21, duration_s=60, n_sources=6)
    monkeypatch.setattr(song, "BUILDUP_P", 1.0)
    plan = plan_song(21, duration_s=60, n_sources=6)
    assert [s["kind"] for s in plan["sources"]] == [s["kind"] for s in base["sources"]]  # same line-up
    step, end = plan["buildup"]["step_bars"], plan["buildup"]["end_bar"]
    entries = sorted(s["entry_bar"] for s in plan["sources"])
    assert entries == [k * step for k in range(len(entries))]
    for s in plan["sources"]:
        assert min(s["bars"]) == s["entry_bar"]
        assert set(range(s["entry_bar"], end)) <= set(s["bars"])  # in until the last entry


def test_build_up_off_by_default():
    assert plan_song(21, duration_s=60)["buildup"] is None


def test_fx_automation_keeps_the_sum(monkeypatch):
    import priism.lab.gen.song as song

    plain_mix, _, plain = render_song(4, duration_s=8, sample_rate=22050, n_sources=4)
    monkeypatch.setattr(song, "FX_AUTO_P", 1.0)
    mix, tracks, meta = render_song(4, duration_s=8, sample_rate=22050, n_sources=4)
    np.testing.assert_allclose(mix, np.sum(tracks, axis=0), atol=1e-5)
    assert [s["kind"] for s in meta["sources"]] == [s["kind"] for s in plain["sources"]]
    for s in meta["sources"]:
        assert len(s["fx_variants"]) >= 2 and s["fx_variants"][0] == s["fx"]
        secs = s["fx_sections"]
        assert secs[0][0] == 0 and all(a[1] == b[0] for a, b in zip(secs, secs[1:]))
        assert all(a[2] != b[2] for a, b in zip(secs, secs[1:]))
    assert not np.allclose(mix, plain_mix)


def test_strict_loops_plan(monkeypatch):
    import priism.lab.gen.song as song

    base = [plan_song(seed, duration_s=30) for seed in range(20)]
    monkeypatch.setattr(song, "LOOP_STRICT_P", 1.0)
    for seed, b in enumerate(base):
        plan = plan_song(seed, duration_s=30)
        assert [s["kind"] for s in plan["sources"]] == [s["kind"] for s in b["sources"]]  # same line-up
        up, n_bars = plan["buildup"], plan["n_bars"]
        loop, step = up["loop_bars"], up["step_bars"]
        assert up["strict"] and step % loop == 0
        assert up["entries"][0] == 0 and all(e % loop == 0 for e in up["entries"])
        for s in plan["sources"]:
            assert s["bars"] == list(range(s["entry_bar"], n_bars))  # in from its entry to the end


def test_strict_loops_off_by_default():
    import priism.lab.gen.song as song

    assert song.LOOP_STRICT_P == 0.0
    assert not any((plan_song(seed, duration_s=30)["buildup"] or {}).get("strict") for seed in range(20))


def test_strict_loops_repeat_exactly(monkeypatch, tmp_path):
    import json

    import priism.lab.gen.song as song

    monkeypatch.setattr(song, "LOOP_STRICT_P", 1.0)
    sr = 22050
    mix, tracks, meta = song.render_song(3, duration_s=20, sample_rate=sr)
    np.testing.assert_allclose(mix, np.sum(tracks, axis=0), atol=1e-5)
    up = meta["buildup"]
    bar, period = up["bar_samples"], up["loop_bars"] * up["bar_samples"]
    assert len(mix) == meta["n_bars"] * bar  # whole samples per bar
    assert len(meta.pop("_master_gain")) == len(mix) // round(song.GAIN_HOP_S * sr)
    assert len(up["master_gain_per_bar"]) == meta["n_bars"]
    p_s = period / sr
    for s in meta["sources"]:  # the notes repeat every loop from the entry on
        t0 = s["entry_bar"] * bar / sr
        first = [e for e in s["timeline"] if e[0] < t0 + p_s]
        nxt = [e for e in s["timeline"] if t0 + p_s <= e[0] < t0 + 2 * p_s]
        if s["entry_bar"] * bar + 2 * period <= len(mix):
            assert len(first) == len(nxt)
            assert all(abs(b[0] - a[0] - p_s) < 2e-4 and a[2:] == b[2:] for a, b in zip(first, nxt))

    # Without the master stage, a part that entered first is periodic once its tails have settled.
    monkeypatch.setattr(song, "_master_gains", lambda m, sr, p: np.ones(len(m)))
    _, tracks, meta = song.render_song(3, duration_s=20, sample_rate=sr)
    for s, t in zip(meta["sources"], tracks):
        if s["entry_bar"] == 0 and len(t) >= 4 * period:
            a, b = t[-2 * period:-period], t[-period:]
            assert np.linalg.norm(b - a) <= 1e-2 * np.linalg.norm(a) + 1e-6

    monkeypatch.undo()
    monkeypatch.setattr(song, "LOOP_STRICT_P", 1.0)
    folder = song.write_song(3, tmp_path, duration_s=8, sample_rate=sr)
    m = json.loads((folder / "meta.json").read_text())
    gain = np.load(folder / m["buildup"]["master_gain_file"])
    assert "_master_gain" not in m and gain["gain"].ndim == 1 and float(gain["hop_s"]) == song.GAIN_HOP_S
