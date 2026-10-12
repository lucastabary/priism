"""The fast registry evaluation (threshold sweeps, tune-then-test) must give exactly the reference results."""

import math

import numpy as np
import pytest
from test_registry import _cached_toys, toy_song

from priism.lab.registry.evaluate import compare
from priism.lab.registry.track import (per_chunk, per_chunk_linked, per_chunk_linked_sweep, per_chunk_sweep,
                                       whole_song, whole_song_sweep)

CUTS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]


@pytest.fixture
def song():
    rng = np.random.default_rng(11)
    emb, energy = toy_song(rng, n_src=5, T=240, entries=[0, 0, 50, 120, 180], drift=0.5, noise=0.3)
    active = energy.sum(0) > 0.5
    active[30:45] = False  # a silent stretch: chunks with no audible token are skipped
    return emb.astype(np.float16), active  # float16, as cached


def test_sweeps_match_one_threshold_runs(song):
    emb, active = song
    for seed in (0, 3):
        got = per_chunk_sweep(emb, active, 20, CUTS, seed)
        linked = per_chunk_linked_sweep(emb, active, 20, CUTS, seed)
        whole = whole_song_sweep(emb, active, CUTS, seed, max_tokens=700, micro=64)  # subsample on
        for c in CUTS:
            np.testing.assert_array_equal(got[c], per_chunk(emb, active, 20, c, seed))
            np.testing.assert_array_equal(linked[c], per_chunk_linked(emb, active, 20, c, seed))
            np.testing.assert_array_equal(whole[c], whole_song(emb, active, c, seed=seed, max_tokens=700, micro=64))


def test_torch_ops_agree_with_numpy(song):
    """The GPU code, run by torch on the CPU: same seeding, float32 sums in another order, so the labels
    may differ only on near ties (none on this well-separated toy)."""
    from priism.lab.registry.track_gpu import gpu_ops

    emb, active = song
    ops = gpu_ops("torch:cpu")
    cpu = whole_song_sweep(emb, active, CUTS, max_tokens=700, micro=64)
    gpu = whole_song_sweep(emb, active, CUTS, max_tokens=700, micro=64, **ops)
    chunks = per_chunk_sweep(emb, active, 20, CUTS, kmeans=ops["kmeans"])
    for c in CUTS:
        assert (cpu[c] == gpu[c]).mean() > 0.99
        assert (chunks[c] == per_chunk(emb, active, 20, c)).mean() > 0.99


def test_whole_song_sweep_handles_silence(song):
    emb, active = song
    none = np.zeros_like(active)
    assert (whole_song_sweep(emb, none, CUTS)[0.5] == -1).all()
    assert (per_chunk_sweep(emb, none, 20, CUTS)[0.5] == -1).all()


def _same(a, b, path=""):
    """Recursive exact equality, NaN == NaN."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), path
    else:
        assert a == b, f"{path}: {a} != {b}"


@pytest.mark.parametrize("level", ["track", "instrument"])
def test_fast_compare_equals_reference(level):
    rng = np.random.default_rng(12)
    tune, test = _cached_toys(rng, 3, T=160, noise=0.4), _cached_toys(rng, 4, T=160, noise=0.4)
    for s in test:  # the grid is keyed by song name: no clash between the sets
        s.name = s.name.replace("toy", "test")
    for s in tune + test:  # instrument level: two of the four sources are one instrument
        for lab in s.labels:
            lab["instrument"] = min(lab["instrument"], 2)
    ref = compare(tune, test, 20, level, log=lambda *_: None, full_grid=True, sweep=False)
    fast = compare(tune, test, 20, level, log=lambda *_: None)
    full = compare(tune, test, 20, level, log=lambda *_: None, full_grid=True)
    _same(ref, full)  # the sweeps change nothing, even on the whole grid
    for key in ("level", "params", "tuning", "n_tune", "n_test", "summary", "tests", "songs", "rows"):
        _same(ref[key], fast[key], key)
    for s in tune:  # tune songs: the whole grid; test songs: only the chosen cells, identical
        _same(ref["grid"][s.name], fast["grid"][s.name], s.name)
    for s in test:
        assert set(fast["grid"][s.name]) == {f"{m}@{g}" for m, g in ref["params"].items()}
        for cell, r in fast["grid"][s.name].items():
            _same(ref["grid"][s.name][cell], r, f"{s.name}/{cell}")
