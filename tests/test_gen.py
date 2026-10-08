import json

import numpy as np
import soundfile as sf

from priism.gen.genres import GENRES
from priism.gen.song import MAX_SOURCES, MIN_SOURCES, plan_song, render_song, write_song


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
    from priism.gen.augment import lossy

    sr = 22050
    t = np.arange(sr * 3) / sr
    x = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)], axis=1).astype(np.float32) * 0.5
    y = lossy(x, sr, "mp3", 128)
    assert y.shape == x.shape
    err = np.sqrt(np.mean((y[sr:2 * sr] - x[sr:2 * sr]) ** 2)) / np.sqrt(np.mean(x**2))
    assert err < 0.2


def test_multi_twins_give_three_or_more_parts_of_one_instrument(monkeypatch):
    from priism.gen import song

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
    from priism.gen import song

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
