import numpy as np

from priism.model.split import split_song

SR = 1000


def fake_model(truth: np.ndarray, starts_seen: list, k: int = 6, seed: int = 0):
    """Returns the true sources of each window, shuffled into k slots, with noisy identity vectors."""
    rng = np.random.default_rng(seed)
    ident = rng.normal(size=(truth.shape[0], 8))
    mix = truth.sum(0)

    def fn(chunk):
        st = next(s for s in range(mix.shape[1]) if np.array_equal(mix[:, s:s + 10], chunk[:, :10]))
        starts_seen.append(st)
        w = chunk.shape[1]
        part = np.zeros((truth.shape[0],) + chunk.shape)
        n = min(w, mix.shape[1] - st)
        part[:, :, :n] = truth[:, :, st:st + n]
        slots = rng.permutation(k)[: truth.shape[0]]
        src = np.zeros((k,) + chunk.shape)
        prob = np.zeros(k)
        att = rng.normal(size=(k, 8))
        for j, s in enumerate(slots):
            src[s] = part[j]
            prob[s] = 0.9
            att[s] = ident[j] + 0.05 * rng.normal(size=8)
        return src, prob, att

    return fn


def snr(ref, est):
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-20))


def test_tracks_keep_their_identity_across_windows_and_pauses():
    rng = np.random.default_rng(1)
    S = 30 * SR
    truth = rng.normal(size=(3, 2, S)) * np.array([1.0, 0.5, 0.25])[:, None, None]
    truth[2, :, 10 * SR:18 * SR] = 0.0  # the third source pauses (a breakdown) then comes back
    seen = []
    tracks, rest, info = split_song(fake_model(truth, seen), truth.sum(0), SR, window_s=4, overlap_s=1)
    assert len(seen) > 5  # several windows
    assert len(tracks) == 3
    for j in range(3):
        assert max(snr(truth[j], t) for t in tracks) > 50
    np.testing.assert_allclose(sum(tracks) + rest, truth.sum(0), atol=1e-5)
    assert np.abs(rest).max() < 1e-5
    assert [i["rms_db"] for i in info] == sorted([i["rms_db"] for i in info], reverse=True)


def test_short_song_is_one_window():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(2, 2, 3 * SR))
    tracks, rest, _ = split_song(fake_model(truth, []), truth.sum(0), SR, window_s=4, overlap_s=1)
    assert len(tracks) == 2 and np.abs(rest).max() < 1e-5
