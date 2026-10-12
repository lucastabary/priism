import json

import numpy as np
import soundfile as sf

from priism.lab.gen import fsloops


def _fake_folder(tmp_path, sr=22050, bpm=120):
    bar = 240 * sr // bpm
    rng = np.random.default_rng(0)
    rows = [("drums", "a", None, 1), ("bass", "b", "A minor", 2), ("pad", "c", "C major", 2),
            ("lead", "d", "F# major", 1), ("percussion", "a", None, 1), ("synth", "e", "A minor", 4)]
    (tmp_path / "audio").mkdir()
    lines = []
    for i, (cat, user, ton, bars) in enumerate(rows):
        x = rng.normal(0, 0.1, (bars * bar, 2)).astype(np.float32)
        sf.write(tmp_path / "audio" / f"{i}.ogg", x, sr)
        lines.append(json.dumps({"id": i, "name": f"l{i}", "username": user, "license": "CC0", "tags": [],
                                 "duration": bars * bar / sr, "bpm": bpm, "tonality": ton,
                                 "tonality_confidence": 0.9, "category": cat}))
    (tmp_path / "index.jsonl").write_text("\n".join(lines) + "\n")
    return bar


def test_keys():
    assert fsloops._compatible((9, "minor"), (0, "major")) and fsloops._compatible((0, "major"), (9, "minor"))
    assert not fsloops._compatible((9, "minor"), (6, "major"))
    assert fsloops._key("Bb minor") == (10, "minor") and fsloops._key(None) is None


def test_arrange_from_loops(tmp_path):
    sr = 22050
    bar = _fake_folder(tmp_path, sr)
    loops = fsloops.load_index(tmp_path)
    assert len(loops) == 6
    for seed in range(5):
        mix, tracks, meta = fsloops.arrange(loops, sr, seed, duration_s=40)
        np.testing.assert_allclose(mix, np.sum(tracks, axis=0), atol=1e-5)
        users = [s["username"] for s in meta["sources"]]
        assert len(users) == len(set(users))  # one loop per uploader
        keys = [fsloops._key(s["tonality"]) for s in meta["sources"] if s["tonality"]]
        assert all(fsloops._compatible(keys[0], k) for k in keys)  # never F# major with A minor
        up = meta["buildup"]
        assert up["bar_samples"] == bar and len(mix) == meta["n_bars"] * bar
        assert all(e % up["loop_bars"] == 0 for e in up["entries"])
        for s, t in zip(meta["sources"], tracks):
            assert not np.any(t[: s["entry_bar"] * bar])
    folder = fsloops.generate(tmp_path, tmp_path / "out", count=1, duration_s=20, sample_rate=sr)[0]
    assert json.loads((tmp_path / "out" / "song_fs000000" / "meta.json").read_text())["genre"] == "freesound_loops"


def test_loop_tempo():
    r = {"name": "Lofi Drum Loop 70 BPM.wav", "tags": [], "duration": 13.71, "bpm": 76}
    assert fsloops.loop_tempo(r) == (70, 4, "name")
    r = {"name": "drums 120 bpm", "tags": [], "duration": 17.05, "bpm": 118}  # 8 bars and a tail
    assert fsloops.loop_tempo(r) == (120, 8, "name")
    r = {"name": "beat", "tags": [], "duration": 1.6, "bpm": 75}  # analysis an octave low: 150 bpm, 1 bar
    assert fsloops.loop_tempo(r) == (150, 1, "analysis x2")
    assert fsloops.loop_tempo({"name": "x", "tags": [], "duration": 5.3, "bpm": 100}) is None
