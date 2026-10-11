import numpy as np
import pytest
import torch

from priism.lab.registry import features as feat
from priism.lab.registry.evaluate import compare, markdown
from priism.lab.registry.head import IdentityHead, supcon, train_head
from priism.lab.registry.metrics import paired_compare, score
from priism.lab.registry.track import online_registry, per_chunk, per_chunk_linked, whole_song


class FakeTrunk:
    """Band log-power per STFT frame as 'features': local, so stitched windows must match one long STFT."""

    sample_rate, hop, n_fft = 8000, 80, 256
    band_bins = np.array([[0, 8], [8, 16], [16, 32], [32, 64], [64, 129]])

    def __call__(self, wav):
        e = feat.band_energy(self, wav.numpy())
        return torch.from_numpy(np.log10(e + 1e-9)[:, :, None].repeat(3, 2))


def test_song_tokens_stitch_windows_like_one_pass():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal((2, 8000 * 7)).astype(np.float32) * np.linspace(0.1, 1, 8000 * 7)
    trunk = FakeTrunk()
    h = feat.song_tokens(trunk, wav, window_s=2.0, margin_s=0.25)
    ref = trunk(torch.from_numpy(wav)).numpy()
    assert h.shape == ref.shape
    np.testing.assert_allclose(h[:-20], ref[:-20], atol=1e-4)  # the zero-padded tail of the last window aside


def test_pool_time_mean_and_sum():
    x = np.arange(10, dtype=np.float32)[:, None]
    np.testing.assert_allclose(feat.pool_time(x, 4)[:, 0], [1.5, 5.5, 8.5])
    np.testing.assert_allclose(feat.pool_time(x, 4, reduce="sum")[:, 0], [6, 22, 17])


def test_projection_keeps_main_directions():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5000, 2)) @ np.array([[3.0, 0, 0, 0], [0, 1.0, 0, 0]]) + 0.01 * rng.standard_normal((5000, 4))
    p = feat.Projection()
    p.update(x)
    p.fit(2)
    assert p.explained > 0.99
    assert p(x).shape == (5000, 2)


def test_cache_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    srcs = rng.standard_normal((3, 2, 8000 * 3)).astype(np.float32)
    song = feat.Song("s", srcs.sum(0), srcs, [{"name": str(j), "instrument": j, "entry_s": 0.0} for j in range(3)])
    trunk = FakeTrunk()
    p = feat.Projection()
    p.update(trunk(torch.from_numpy(song.mix)).numpy())
    p.fit(2)
    c = feat.song_cache(trunk, song, p, pool=4, window_s=2.0, margin_s=0.25)
    feat.write_cache(tmp_path / "s.npz", c)
    got = feat.load_cache(tmp_path / "s.npz")
    T = -(-(8000 * 3 // 80 + 1) // 4)
    assert got.tokens.shape == (T, 5, 2) and got.energy.shape == (3, T, 5)
    assert list(got.instruments) == [0, 1, 2]
    assert got.frame_s == pytest.approx(0.04)


# ------------------------------------------------------------------------------------------- synthetic songs

def toy_song(rng, n_src=4, T=200, Nb=8, d=16, entries=None, drift=0.0, noise=0.15):
    """Embeddings where each source has its own direction and owns some tokens; sources enter at ``entries``."""
    dirs = feat_norm(rng.standard_normal((n_src, d)))
    entries = entries if entries is not None else [0] * n_src
    energy = np.full((n_src, T, Nb), 1e-12)
    owner = rng.integers(n_src, size=(T, Nb))
    for t in range(T):
        live = [j for j in range(n_src) if entries[j] <= t]
        owner[t] = rng.choice(live, size=Nb)
    for j in range(n_src):
        energy[j][owner == j] = 1.0
    shift = drift * np.linspace(0, 1, T)[:, None, None] * rng.standard_normal((1, 1, d))
    emb = dirs[owner] + shift + noise * rng.standard_normal((T, Nb, d))
    return emb.astype(np.float32), energy.astype(np.float32)


def feat_norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_perfect_labels_score_one_and_permuted_chunks_are_caught():
    rng = np.random.default_rng(3)
    _, energy = toy_song(rng)
    owner = energy.argmax(0)
    s = score(owner, energy, chunk=20)
    assert s["purity"] == pytest.approx(1) and s["completeness"] == pytest.approx(1)
    assert s["permutation_rate"] == 0 and s["track_loss"] == pytest.approx(0)
    swapped = owner.copy()
    swapped[100:] = (owner[100:] + 1) % 4  # ids rotated in the second half: clean chunks, lost tracks
    s2 = score(swapped, energy, chunk=20)
    assert s2["local_completeness"] == pytest.approx(1)
    assert s2["completeness"] < 0.6 and s2["permutation_rate"] > 0.4 and s2["track_loss"] > 0.4


def test_whole_song_and_registry_beat_unlinked_chunks():
    rng = np.random.default_rng(4)
    emb, energy = toy_song(rng, entries=[0, 0, 60, 120])
    active = energy.sum(0) > 0.5
    chunk = 20
    s_chunk = score(per_chunk(emb, active, chunk, 0.5), energy, chunk)
    s_link = score(per_chunk_linked(emb, active, chunk, 0.5), energy, chunk)
    s_song = score(whole_song(emb, active, 0.5), energy, chunk)
    lab, births, reg = online_registry(emb, active, chunk, tau=0.5)
    s_reg = score(lab, energy, chunk, births=births)
    assert s_chunk["completeness"] < 0.5
    for s in (s_link, s_song, s_reg):
        assert s["purity"] > 0.95 and s["completeness"] > 0.95
    assert len(reg) == 4
    assert s_reg["entry_f1"] == pytest.approx(1)  # births at 0, 0, 60, 120, each the right source


def test_paired_compare_detects_a_clear_gain():
    rng = np.random.default_rng(5)
    b = rng.uniform(0.4, 0.6, 30)
    c = paired_compare(b + 0.1 + 0.01 * rng.standard_normal(30), b)
    assert c["ci95"][0] > 0.05 and c["wilcoxon_p"] < 1e-3 and c["d_z"] > 2


def _cached_toys(rng, n, **kw):
    out = []
    for i in range(n):
        emb, energy = toy_song(rng, **kw)
        labels = [{"name": str(j), "instrument": j, "entry_s": 0.0} for j in range(energy.shape[0])]
        out.append(feat.Cached(f"toy{i}", emb, energy, labels, 0.04))
    return out


def test_compare_runs_end_to_end():
    rng = np.random.default_rng(6)
    tune, test = _cached_toys(rng, 3), _cached_toys(rng, 4)
    r = compare(tune, test, chunk=20, level="track", log=lambda *_: None)
    assert set(r["summary"]) == {"per_chunk", "per_chunk_linked", "whole_song", "online", "online_two_pass"}
    assert r["summary"]["whole_song"]["completeness"]["mean"] > 0.9
    assert "online_two_pass - per_chunk / completeness" in r["tests"]
    assert "| method |" in markdown(r)


def test_supcon_prefers_grouped_embeddings():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1, 0.1], [0, 1], [0.1, 1]]), dim=1)
    bad = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [0, 1], [1, 0.1], [0.1, 1]]), dim=1)
    assert supcon(good, labels) < supcon(bad, labels)


def test_identity_head_learns_and_guards(tmp_path):
    rng = np.random.default_rng(7)
    # Sources whose raw tokens drift over the song: the head must learn what stays the same.
    train = _cached_toys(rng, 6, T=300, d=8, drift=1.0, noise=0.3)
    valid = _cached_toys(rng, 2, T=300, d=8, drift=1.0, noise=0.3)
    head = IdentityHead(8, 8, dim=8, hidden=32, context=1)
    assert head(torch.from_numpy(train[0].tokens)).shape == (300, 8, 8)
    s = train_head(train, valid, tmp_path, level="track", steps=60, songs_per_step=3, per_ref=8, chunk_s=0.8,
                   min_gap_s=4.0, eval_every=20, dim=8, context=1, plateau=0, log=lambda *_: None)
    assert s["stopped"] == "steps"
    hist = [__import__("json").loads(x) for x in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert len(hist) == 3 and hist[-1]["loss"] < hist[0]["loss"]
    assert (tmp_path / "best.pt").exists()
