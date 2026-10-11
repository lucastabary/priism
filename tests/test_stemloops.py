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
