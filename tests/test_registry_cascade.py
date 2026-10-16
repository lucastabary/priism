import json

import numpy as np
import torch

from priism.lab.registry import cascade as cas
from priism.lab.registry import separate_eval as ev
from test_registry_separate import SR, FakeStems, _proj, _toy


class BandSplitter:
    """A fake specialist: the stem cut by FFT bins into 'mid' (< 1 kHz) and 'top' (1-2 kHz); the rest of the
    spectrum (> 2 kHz) is left to the step's rest."""

    stems = ("mid", "top")
    sample_rate = SR
    name = "band"

    def __call__(self, wav):
        X = np.fft.rfft(wav, axis=-1)
        f = np.fft.rfftfreq(wav.shape[-1], 1 / SR)
        mid = np.fft.irfft(X * (f < 1000), wav.shape[-1], axis=-1)
        top = np.fft.irfft(X * ((f >= 1000) & (f < 2000)), wav.shape[-1], axis=-1)
        return np.stack([mid, top]).astype(np.float32)


def test_split_sums_to_the_stem_in_every_variant():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3001)).astype(np.float32)
    step = cas.CascadeStep("high", BandSplitter(), {"both": ["mid", "top"]})
    out = cas.split(step, x, SR)
    assert out[False][0] == ["high_mid", "high_top", "high_rest"]
    assert out[True][0] == ["high_both", "high_rest"]
    for names, ys in out.values():
        assert all(y.shape == x.shape for y in ys)
        np.testing.assert_allclose(np.sum(ys, 0), x, atol=1e-5)


def test_split_resamples_to_the_splitter_rate_and_back():
    class Half(BandSplitter):
        sample_rate = SR // 2

    x = np.random.default_rng(1).standard_normal((2, 2001)).astype(np.float32)
    names, ys = cas.split(cas.CascadeStep("high", Half()), x, SR)[False]
    assert len(ys) == 3 and all(y.shape == x.shape for y in ys)
    np.testing.assert_allclose(np.sum(ys, 0), x, atol=1e-5)  # the rest absorbs what resampling lost


def test_chunked_overlap_add_is_exact_for_an_identity_model():
    for S in (50, 999, 1000, 4321):
        x = torch.randn(2, S)
        y = cas.chunked(lambda b: b[:, None].repeat(1, 2, 1, 1), x, 2, chunk=400, overlap=4, batch=3)
        assert y.shape == (2, 2, S)
        torch.testing.assert_close(y[0], x, atol=1e-5, rtol=1e-5)


def test_cascade_systems_in_the_eval():
    song = _toy(SR * 4)  # bass tone in 'low'; two guitars (bins 48 = 1.5 kHz and 96 = 3 kHz) in 'high'
    steps = [cas.CascadeStep("high", BandSplitter(), {"both": ["mid", "top"]})]
    rec = ev.evaluate_song(FakeStems(), song, _proj(), None, cuts=(0.5,), window_s=2.0, margin_s=0.25,
                           oracle_bin=False, cascade=steps)
    json.dumps(rec)
    names = set(rec["refs"][0])
    assert {"cascade", "cascade_m2o", "cascade_facet", "cascade_merged", "cascade_merged_facet"} <= names
    assert rec["n_outputs"]["cascade"] == 1 + 3 and rec["n_outputs"]["cascade_merged"] == 1 + 2
    guitars = [r for r in rec["refs"] if r["stem"] == "guitar"]
    # the specialist separates the two guitars (1.5 kHz in 'top', 3 kHz in the rest); SW cannot
    assert all(r["cascade"]["sdr"] > r["sw"]["sdr"] + 10 for r in guitars)
    summary = ev.summarize([rec, {**rec, "song": "toy2", "artist": "other"}])
    for k in ("cascade_facet - registry (all", "cascade - sw (all", "cascade_facet - cascade (all",
              "cascade_merged - cascade (all"):
        assert any(p.startswith(k) for p in summary["paired"]), k
    assert ev.table(summary)
