import json

import numpy as np
import soundfile as sf
import torch

from priism.lab.registry import features as feat
from priism.lab.registry import separate as sep
from priism.lab.registry import separate_eval as ev
from priism.lab.registry.head import IdentityHead

SR, HOP, NFFT = 8000, 80, 256
BANDS = np.array([[0, 8], [8, 16], [16, 32], [32, 64], [64, 129]])


class FakeStems:
    """Two 'stems' split by fixed bins (low: bins < 32, high: the rest); taps = band log power (3 dims)."""

    sample_rate, hop, n_fft, band_bins, stems = SR, HOP, NFFT, BANDS, ("low", "high")

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(wav, NFFT, HOP, window=torch.hann_window(NFFT), center=True, return_complex=True)

    def __call__(self, wav):
        spec = self._stft(wav)  # (C, F, T)
        m = torch.zeros(2, spec.shape[1], 1)
        m[0, :32], m[1, 32:] = 1, 1
        e = feat.band_energy(self, wav.numpy())
        h = np.log10(e + 1e-9)[:, :, None].repeat(3, 2).astype(np.float32)
        return h, (spec[None] * m[:, None]).numpy().astype(np.complex64)

    def istft(self, spec, length):
        x = torch.from_numpy(np.ascontiguousarray(spec))
        lead = x.shape[:-2]
        y = torch.istft(x.reshape(-1, *x.shape[-2:]), NFFT, HOP, window=torch.hann_window(NFFT), center=True,
                        length=length)
        return y.reshape(*lead, length).numpy()


def _proj():
    return feat.Projection(np.zeros(3, np.float32), np.eye(3, dtype=np.float32)[:, :2])


def _tone(bin_: float, n: int, amp: float = 0.5, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * bin_ * SR / NFFT * t + phase)
    return np.stack([x, 0.8 * x]).astype(np.float32)


def _toy(n: int = SR * 5):
    """Three references: one in the low stem (bin 10), two in the high stem (bins 48 and 96, other bands)."""
    refs = np.stack([_tone(10, n), _tone(48, n, 0.4), _tone(96, n, 0.3, 1.0)])
    labels = [{"name": f"r{j}", "stem": s, "kind": s, "instrument": j, "group_key": "artist"}
              for j, s in enumerate(["bass", "guitar", "guitar"])]
    return feat.Song("toy", refs.sum(0), refs, labels)


def test_window_plan_covers_every_frame_once():
    for n in (1, 50, 401, 1000, 1234):
        plan = sep.window_plan(n, HOP, SR, 2.0, 0.25)
        frames = np.concatenate([np.arange(s + lo, s + hi) for s, lo, hi in plan])
        np.testing.assert_array_equal(frames, np.arange(n))


def test_analysis_shapes_and_tokens_match_the_cache():
    rng = np.random.default_rng(0)
    S = SR * 7 + 37
    wav = (rng.standard_normal((2, S)) * np.linspace(0.1, 1, S)).astype(np.float32)
    model = FakeStems()
    an = sep.analyse(model, wav, _proj(), pool=4, window_s=2.0, margin_s=0.25)
    T = S // HOP + 1
    assert an.spec.shape == (2, 2, NFFT // 2 + 1, T)
    assert an.tokens.shape == (-(-T // 4), len(BANDS), 2) and an.tokens.dtype == np.float16
    assert an.energy.shape == (2, -(-T // 4), len(BANDS))
    # the cache's path: song_tokens (one stitched pass), PCA, pool
    cache = feat.pool_time(_proj()(_cache_tokens(model, wav)), 4).astype(np.float16)
    np.testing.assert_allclose(an.tokens.astype(np.float32), cache.astype(np.float32), atol=2e-2)


def _cache_tokens(model, wav):
    class T:
        sample_rate, hop, n_fft, band_bins = SR, HOP, NFFT, BANDS

        def __call__(self, w):
            return torch.from_numpy(model(w)[0])

    return feat.song_tokens(T(), wav, 2.0, 0.25)


def test_registry_split_sums_to_stem_and_keeps_exact_length(tmp_path):
    song = _toy(SR * 6 + 11)
    model = FakeStems()
    an = sep.analyse(model, song.mix, _proj(), window_s=2.0, margin_s=0.25)
    head = IdentityHead(2, len(BANDS), dim=8, hidden=16, context=1).eval()
    emb = sep.embed(an.tokens, head)
    w = sep.registry_weights(emb, an.energy, [0.05, 0.5], own=0.5, min_tokens=8)
    for cut, per_stem in w.items():
        for s, (weights, info) in enumerate(per_stem):
            np.testing.assert_allclose(weights.sum(0), 1, atol=1e-5)
            assert len(info) == len(weights)
            stem = model.istft(an.spec[s], an.length)
            outs = list(sep.render(model, an.spec[s], weights, an.pool, an.length, smooth=2))
            assert all(o.shape == (2, song.mix.shape[1]) for o in outs)
            np.testing.assert_allclose(np.sum(outs, 0), stem, atol=1e-4)
    # soft weights are a partition of unity too
    soft = sep.registry_weights(emb, an.energy, [0.3], temp=0.1, min_tokens=8)[0.3]
    for weights, _ in soft:
        np.testing.assert_allclose(weights.sum(0), 1, atol=1e-5)


def test_smooth_time_keeps_partition():
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 3, (40, 5))
    w = np.eye(3, dtype=np.float32)[lab].transpose(2, 0, 1)
    np.testing.assert_allclose(sep.smooth_time(w, 3).sum(0), 1, atol=1e-6)


def test_oracle_labels_split_the_toy_perfectly():
    song = _toy()
    model = FakeStems()
    an = sep.analyse(model, song.mix, _proj(), window_s=2.0, margin_s=0.25)
    ow = sep.oracle_weights(sep.ref_token_energy(model, song.sources, an.pool), np.array([0, 1, 1]), 2)
    assert ow[0][1] == [0] and ow[1][1] == [1, 2]
    outs = [list(sep.render(model, an.spec[s], ow[s][0], an.pool, an.length)) for s in range(2)]
    edge = slice(NFFT, -NFFT)  # the zero-padded tail of the last window aside
    assert ev.sdr(song.sources[0][:, edge], outs[0][0][:, edge]) > 30
    assert ev.sdr(song.sources[1][:, edge], outs[1][0][:, edge]) > 30
    assert ev.sdr(song.sources[2][:, edge], outs[1][1][:, edge]) > 30
    # SW alone gives the two guitars one stem: neither is separated
    stem = model.istft(an.spec[1], an.length)
    assert ev.sdr(song.sources[1], stem) < 6


def test_sdr_definitions():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((2, 10000))
    n = rng.standard_normal((2, 10000))
    n *= np.sqrt((s ** 2).sum() / (n ** 2).sum() / 100)  # signal / noise = 100: 20 dB
    assert abs(ev.sdr(s, s + n) - 20) < 1e-6
    assert abs(ev.sdr(s, 2 * s)) < 1e-6  # |s|^2 / |s - 2s|^2 = 1
    assert ev.si_sdr(s, 2 * s) > 100  # scale does not matter
    assert abs(ev.si_sdr(s, 3 * (s + n)) - ev.si_sdr(s, s + n)) < 1e-6
    assert abs(ev.sdr(s, np.zeros_like(s))) < 1e-6  # a silent estimate: 0 dB
    # the streamed statistics give the same numbers as the direct definitions
    refs = np.stack([s, n]).astype(np.float32)
    ests = [s + n, 0.5 * s, n - s]
    stats = [ev.output_stats(refs, e.astype(np.float32), block=777) for e in ests]
    S, SI = ev.score_matrix(ev.ref_energies(refs, block=999), np.array([d for d, _ in stats]),
                            np.array([e for _, e in stats]))
    for o, e in enumerate(ests):
        for j in range(2):
            assert abs(S[o, j] - ev.sdr(refs[j], e.astype(np.float32))) < 1e-4
            si = ev.si_sdr(refs[j], e.astype(np.float32))
            assert abs(SI[o, j] - si) < 1e-4 * max(1.0, abs(si))  # 117 dB for 0.5 s: float64 cancellation


def test_matching_one_to_one_and_many_to_one():
    S = np.array([[10.0, 9.0, 1.0], [2.0, 3.0, 0.0]])
    np.testing.assert_array_equal(ev.match(S, False), [0, 0, 0])
    o2o = ev.match(S, True)
    assert sorted(o2o[o2o >= 0].tolist()) == [0, 1] and (o2o == -1).sum() == 1
    s, _ = ev.picked(S, S, o2o)
    assert s[o2o == -1][0] == 0.0  # no output left: scored as silence


def test_evaluate_song_and_summary(tmp_path):
    song = _toy(SR * 4)
    model = FakeStems()
    rec = ev.evaluate_song(model, song, _proj(), None, cuts=(0.1, 0.5), window_s=2.0, margin_s=0.25)
    json.dumps(rec)
    assert rec["n_scored"] == 3 and {"sw", "sw_1to1", "registry@0.1", "registry@0.5_m2o", "oracle"} <= set(rec["refs"][0])
    guitars = [r for r in rec["refs"] if r["stem"] == "guitar"]
    assert all(r["shared"] for r in guitars)
    assert all(r["oracle"]["sdr"] > r["sw"]["sdr"] + 10 for r in guitars)
    summary = ev.summarize([rec, {**rec, "song": "toy2", "artist": "other"}])
    assert summary["systems"]["oracle"]["sdr"]["shared"]["n"] == 4
    assert "by_kind" in summary["systems"]["sw"]["sdr"]["all"]
    assert ev.table(summary)


def test_oracle_count_system(tmp_path):
    rec = ev.evaluate_song(FakeStems(), _toy(SR * 4), _proj(), None, cuts=(0.5,), window_s=2.0, margin_s=0.25,
                           oracle_k=True)
    assert {"registry_k", "registry_k_m2o"} <= set(rec["refs"][0]) and rec["n_outputs"]["registry_k"] >= rec["n_outputs"]["sw"]
    summary = ev.summarize([rec, {**rec, "song": "toy2", "artist": "other"}])
    assert any(k.startswith("registry_k - registry") for k in summary["paired"])


def test_separate_file_writes_exact_length_at_input_rate(tmp_path):
    song = _toy(SR * 3 + 5)
    path = tmp_path / "in.wav"
    sf.write(path, song.mix.T, 16000)  # another rate than the model's: resampled in and out
    meta = sep.separate_file(FakeStems(), path, tmp_path / "out", _proj(), None, cut=0.3, window_s=2.0,
                             margin_s=0.25, write_stems=True, log=lambda *a: None)
    assert (tmp_path / "out" / "sources.json").exists()
    total = 0
    for s in meta["sources"]:
        y, sr = sf.read(tmp_path / "out" / s["file"], always_2d=True)
        assert sr == 16000 and y.shape == (song.mix.shape[1], 2)
        total = total + y
    stems = sum(sf.read(tmp_path / "out" / "sw" / f"{n}.wav", always_2d=True)[0] for n in ("low", "high"))
    np.testing.assert_allclose(total, stems, atol=1e-4)
