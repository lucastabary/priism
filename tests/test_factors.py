import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from priism.gen.song import render_song, write_song  # noqa: E402
from priism.model.data import SongChunks, collate  # noqa: E402
from priism.model.factors import FactorHeads, factor_loss, harmonic_comb, rasterize  # noqa: E402


def test_every_source_has_a_timeline_that_matches_its_audio():
    sr = 22050
    for seed in range(6):
        _, tracks, meta = render_song(seed, duration_s=10.0, sample_rate=sr)
        for s, t in zip(meta["sources"], tracks):
            tl = np.asarray(s["timeline"]).reshape(-1, 5)
            assert len(tl), s["kind"]
            hop = 512
            frames = len(t) // hop
            act = rasterize(tl, frames, hop / sr)["active"].numpy() > 0
            e = np.array([np.mean(t[i * hop:(i + 1) * hop] ** 2) for i in range(frames)])
            if act.mean() > 0.9 or not act.any():
                continue
            # Loudness follows the notes better than it follows the same timeline moved by an 8th note
            # (pads swell and effects ring on, so "louder while playing" alone is not a fair check).
            late = tl.copy()
            late[:, :2] += 30.0 / meta["bpm"]
            moved = rasterize(late, frames, hop / sr)["active"].numpy() > 0
            assert np.corrcoef(act, e)[0, 1] > np.corrcoef(moved, e)[0, 1], (seed, s["kind"])


def test_rasterize_marks_notes_onsets_and_chords():
    tl = np.array([[0.10, 0.30, 60, 0.8, 1], [0.10, 0.30, 64, 0.8, 1], [0.30, 0.50, 62, 0.5, 0], [0.6, 0.7, -1, 1, 1]])
    r = rasterize(tl, 80, 0.01)
    assert r["roll"][20, 60] == 1 and r["roll"][20, 64] == 1 and r["roll"][20].sum() == 2
    assert r["roll"][40, 62] == 1 and r["roll"][65].sum() == 0  # the hit is unpitched
    assert r["onset"][10] == 1 and r["onset"][30] == 0 and r["onset"][60] == 1  # a slide has no attack
    assert r["active"][65] == 1 and r["active"][55] == 0
    assert abs(float(r["vel"][40]) - 0.5) < 1e-6


def test_harmonic_comb_puts_a_low_note_in_low_bands():
    edges = torch.arange(0, 11025 + 1, 100.0)
    comb = harmonic_comb(torch.stack([edges[:-1], edges[1:]], 1))
    sums = comb.sum(1)
    assert ((sums - 1).abs() < 1e-5).logical_or(sums == 0).all()  # notes above Nyquist have no band
    assert comb[45].argmax() == 1  # A2 = 110 Hz: its fundamental, the strongest share, is in 100-200 Hz


def test_crops_carry_the_notes_of_each_heard_source(tmp_path):
    write_song(3, tmp_path, duration_s=8.0, sample_rate=22050)
    ds = SongChunks(tmp_path, 2.0, 22050, labels=True)
    mix, tg, lab = ds[0]
    assert len(lab["timeline"]) == len(tg) == len(lab["families"]) == len(lab["fx"])
    for tl in lab["timeline"]:
        assert tl.shape[1] == 5 and (tl[:, 1] > 0).all() and (tl[:, 0] < 2.0).all()
    mixes, tgs, n, labels = collate([ds[0], ds[1]])
    assert mixes.shape[0] == 2 and len(labels) == 2


def test_factor_loss_learns_and_z_finds_its_source():
    torch.manual_seed(0)
    bands = torch.tensor([[i * 200.0, (i + 1) * 200.0] for i in range(12)])
    heads = FactorHeads(16, bands, hidden=32)
    x = torch.randn(4, 20, 12, 16)
    f = heads(x)
    assert f["pitch"].shape == (4, 20, 128) and f["z"].shape == (4, 64) and f["z_halves"].shape == (4, 2, 64)
    tl = [np.array([[0.0, 0.2, 48 + 5 * k, 0.9, 1]]) for k in range(2)]
    labels = [{"timeline": tl, "families": [0, 1], "fx": np.zeros((2, 11), np.float32)}] * 2
    match = [(np.array([0, 1]), np.array([0, 1])), (np.array([1, 0]), np.array([0, 1]))]
    opt = torch.optim.Adam(heads.parameters(), 1e-2)
    first = None
    for _ in range(60):
        loss, st = factor_loss(heads(x), labels, match, K=2, frame_s=0.01)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first and st["pitch_f1"] > 0.6 and st["z_acc"] == 1.0


def test_tiny_training_with_factors_logs_their_scores_and_reloads(tmp_path):
    from priism.model.train import load_run, train

    write_song(5, tmp_path / "songs", duration_s=6.0, sample_rate=22050)
    h = train(tmp_path / "songs", tmp_path / "run", preset="tiny", steps=2, batch=2, chunk_s=1.0, log_every=1,
              factors=True, factor_feedback=True, log=lambda *_: None)
    assert "pitch_f1" in h[-1] and "z_acc" in h[-1]
    model, _ = load_run(tmp_path / "run")
    assert hasattr(model, "factors") and model.factors.feedback


def _small_msst(**kw):
    msst = os.environ.get("MSST_PATH")
    if not msst:
        pytest.skip("MSST_PATH not set (MSST checkout needed)")
    sys.path.insert(0, msst)
    pytest.importorskip("rotary_embedding_torch")
    from models.bs_roformer.bs_roformer import BSRoformer

    from priism.model.msst_core import MsstAttractorSeparator

    bands = (2,) * 24 + (4,) * 8 + (8,) * 4 + (17,)
    torch.manual_seed(0)
    r = BSRoformer(dim=16, depth=1, stereo=True, num_stems=4, freqs_per_bands=bands, dim_head=8, heads=2,
                   stft_n_fft=256, stft_hop_length=64, stft_win_length=256, flash_attn=False)
    return MsstAttractorSeparator(r, max_sources=5, decoder_depth=1, heads=2, **kw)


def test_msst_factors_start_where_a_run_without_them_was():
    base = _small_msst(v2=True).eval()
    fb = _small_msst(v2=True, factor_feedback=True).eval()
    missing, unexpected = fb.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("factors.") for k in missing)
    mix = torch.randn(2, 2, 4096) * 0.1
    with torch.no_grad():
        a, b = base(mix), fb(mix)
    torch.testing.assert_close(a["sources"], b["sources"])  # the feedback starts at zero
    assert b["factors"]["pitch"].shape[:2] == (10, a["sources"].shape[-1] // 64 + 1)
    fb.train()
    o = fb(mix)
    (o["sources"].abs().mean() + o["factors"]["pitch"].mean()).backward()
    assert fb.factors.back.weight.grad is not None and fb.factors.pitch.weight.grad is not None
    assert fb._band_hz(44100)[-1, 1] == pytest.approx(44100 / 2 + 44100 / 256)
