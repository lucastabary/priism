import numpy as np
import pytest

from priism.lab.gen import stemloops


def _fake_stems(sr: int, bar: int, n_bars: int, rng: np.random.Generator) -> np.ndarray:
    """4 stems (4, 2, S): a looping click track, a looping tone, a looping noise burst, one non-looping noise."""
    n = n_bars * bar
    t = np.arange(n) / sr
    clicks = np.zeros(n)
    for k in range(0, n, bar // 4):
        clicks[k: k + 200] = np.hanning(200)
    tone = np.sin(2 * np.pi * 110 * t) * ((np.arange(n) % bar) < bar // 2)
    burst = np.tile(rng.normal(0, 0.1, bar), n_bars) * ((np.arange(n) % bar) > 3 * bar // 4)
    free = rng.normal(0, 0.1, n)  # never repeats: not a loop
    return np.stack([np.stack([x, x]) for x in (clicks, tone, burst, free)]).astype(np.float32)


def test_arrange_loops_real_stems():
    sr, bar = 8000, 16000  # 120 bpm
    stems = _fake_stems(sr, bar, 24, np.random.default_rng(0))
    assert abs(stemloops.bar_period(stems.sum(0).mean(0), sr) - bar) <= 2 or \
        abs(stemloops.bar_period(stems.sum(0).mean(0), sr) - bar // 2) <= 2
    windows = stemloops.loop_windows(stems, bar, loops=(1, 2))
    assert windows[0]["looping"] == 3 and windows[0]["sdr_db"][3] < stemloops.LOOP_SDR_DB
    mix, tracks, meta = stemloops.arrange(stems, sr, seed=1, duration_s=20, windows=windows, bar=bar)
    np.testing.assert_allclose(mix, np.sum(tracks, axis=0), atol=1e-5)
    up = meta["buildup"]
    assert len(tracks) == 3 and {s["stem"] for s in meta["sources"]} == {0, 1, 2}  # the free stem is left out
    assert all(s["kind"] == "stem_group" for s in meta["sources"])
    assert len(mix) == meta["n_bars"] * bar and up["entries"][0] == 0
    period = up["loop_bars"] * bar
    g = np.repeat(meta["_master_gain"], round(stemloops.GAIN_HOP_S * sr))
    g = np.concatenate([g, np.full(len(mix) - len(g), g[-1])])
    for s, t in zip(meta["sources"], tracks):
        assert s["bars"] == list(range(s["entry_bar"], meta["n_bars"]))
        before = t[: max(0, s["entry_bar"] * bar - int(stemloops.FADE_S * sr) - 1)]
        assert np.all(before == 0)  # silent before its entry
        # Exactly periodic after the entry once the master gain is undone (hop-mean gain: loose tolerance).
        a, b = t[-2 * period:-period] / g[-2 * period:-period, None], t[-period:] / g[-period:, None]
        assert np.linalg.norm(b - a) <= 0.05 * np.linalg.norm(a)


def test_only_the_train_part():
    with pytest.raises(ValueError):
        stemloops.generate("nowhere", "nowhere", part="test")


def test_entry_quality_on_exact_loops():
    from priism.lab.gen.song import entry_quality

    sr, bar = 8000, 16000
    stems = _fake_stems(sr, bar, 24, np.random.default_rng(0))
    mix, tracks, meta = stemloops.arrange(stems, sr, seed=3, duration_s=40, bar=bar)
    rows = meta["buildup"]["entry_quality"]
    assert [r["bar"] for r in rows] == meta["buildup"]["entries"][1:len(rows) + 1]
    assert all(r["diff_sdr_db"] is not None for r in rows)
    # Recomputed from the mix alone, the drift is the same (it never needs the stems).
    again = entry_quality(mix, meta["buildup"])
    assert [r["drift_db"] for r in again] == [r["drift_db"] for r in rows]


def test_unpitched_stems_take_their_own_window():
    sr, bar, n_bars = 8000, 16000, 24
    rng = np.random.default_rng(1)
    stems = _fake_stems(sr, bar, n_bars, rng)
    half = n_bars // 2 * bar
    stems[0, :, half:] = rng.normal(0, 0.1, (2, stems.shape[2] - half))  # drums loop in the first half only
    stems[1, :, :half] = rng.normal(0, 0.1, (2, half))  # the tone loops in the second half only
    stems[2] = 0.0
    names = ["Drums", "Bass", "Synths", "Vox"]
    windows = stemloops.loop_windows(stems, bar, loops=(1,))
    assert not stemloops.plans(windows, set())  # one shared window never holds two looping stems
    mix, tracks, meta = stemloops.arrange(stems, sr, seed=0, duration_s=20, windows=windows, bar=bar, names=names)
    by = {s["stem_name"]: s for s in meta["sources"]}
    assert set(by) == {"Drums", "Bass"} and by["Drums"]["own_window"] and not by["Bass"]["own_window"]
    assert by["Drums"]["window_start_s"] * sr < half <= by["Bass"]["window_start_s"] * sr
    assert stemloops.UNPITCHED.match("Perc") and not stemloops.UNPITCHED.match("808")


def test_multitrack_loops_train_artists_only(tmp_path):
    import json

    import soundfile as sf

    from priism.lab.registry.run import in_part

    sr, bar = 8000, 16000
    artists = [a for a in (f"artist{i}" for i in range(40)) if in_part(a, "train")][:1] + \
              [a for a in (f"artist{i}" for i in range(40)) if in_part(a, "test")][:1]
    for n, artist in enumerate(artists):
        song = tmp_path / "db" / f"song{n}"
        stems = _fake_stems(sr, bar, 24, np.random.default_rng(n))
        layout = [("drums", "kick_drum"), ("bass", "synth_bass"), ("other_keys", "synth_lead"), ("vocals", "lead_male_singer")]
        meta = {"artist": artist, "genre": "electronic", "song": f"s{n}", "stems": []}
        for j, (stem, ttype) in enumerate(layout):
            (song / stem).mkdir(parents=True)
            sf.write(song / stem / f"t{j}.flac", stems[j].T, sr)
            meta["stems"].append({"id": f"st{j}", "stemName": stem,
                                  "tracks": [{"id": f"t{j}", "trackType": ttype, "has_bleed": False}]})
        (song / "data.json").write_text(json.dumps(meta))
    out = stemloops.generate_multitrack(tmp_path / "db", tmp_path / "out", per_song=1, duration_s=20,
                                        sample_rate=sr)
    assert [p.split("\\")[-1].split("/")[-1] for p in out] == ["song0__loop0"]  # the test artist is never read
    m = json.loads((tmp_path / "out" / "song0__loop0" / "meta.json").read_text())
    kinds = {s["kind"] for s in m["sources"]}
    assert kinds <= {"kick_drum", "synth_bass", "synth_lead"} and len(kinds) >= 2  # the noise "vocals" never loop
    assert all(s["artist"] == artists[0] and "has_bleed" in s for s in m["sources"])
