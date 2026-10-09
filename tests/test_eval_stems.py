import numpy as np
import soundfile as sf

from priism.model.eval_stems import busiest, evaluate, score_excerpt


def test_score_excerpt_strict_vs_grouped():
    rng = np.random.default_rng(0)
    refs = rng.standard_normal((3, 2, 1000)).astype(np.float32)
    groups = np.zeros(3, bool)
    # Perfect split: both scores high for every instrument.
    s, g = score_excerpt(refs.copy(), refs, groups)
    assert min(s) > 30 and min(g) > 30
    # Two instruments merged in one output: grouped still credits the louder share, strict leaves one at 0 dB.
    est = np.stack([refs[0] + refs[1], refs[2]])
    s, g = score_excerpt(est, refs, groups)
    assert min(s) <= 0.5 and s[2] > 30


def test_busiest_prefers_full_arrangement():
    stems = np.zeros((3, 2, 4000), np.float32)
    stems[0] = 1.0
    stems[1:, :, 2000:] = 1.0  # two more instruments enter halfway
    starts = busiest(stems, 1000, 2, level=float(np.mean(stems.sum(0) ** 2)))
    assert all(st > 1000 for st in starts)  # every pick overlaps the full arrangement


def test_evaluate_on_mshoxx_layout(tmp_path):
    rng = np.random.default_rng(1)
    song = tmp_path / "s"
    song.mkdir()
    stems = {n: 0.1 * rng.standard_normal(44100 * 2).astype(np.float32) for n in ("drums", "bass", "mel1")}
    for n, x in stems.items():
        sf.write(song / f"s_{n}.flac", x, 44100)
    oracle = lambda mix: np.stack([np.repeat(x[None, : mix.shape[1]], 2, 0) for x in stems.values()])
    summ = evaluate(oracle, tmp_path, tmp_path / "out", segment_s=2.0, segments=1, log=lambda *_: None)
    assert summ["strict_above_5db"] == 1.0 and summ["drums_mean"] > 30


def test_cascade_splits_loud_stems_only():
    from priism.model.eval_stems import cascade_fn

    mix = np.ones((2, 100), np.float32)
    first = lambda m: np.stack([m * 0.999, m * 0.001])  # second stem is 60 dB down: kept whole
    second = lambda s: np.stack([s * 0.5, s * 0.5])
    out = cascade_fn(first, second)(mix)
    assert out.shape == (3, 2, 100) and np.allclose(out.sum(0), mix)
