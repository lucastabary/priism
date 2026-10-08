import pytest

torch = pytest.importorskip("torch")

from priism.model.data import SongChunks, collate, merged_sources  # noqa: E402
from priism.model.loss import neg_snr, pit_loss  # noqa: E402
from priism.model.separator import AttractorSeparator, default_bands  # noqa: E402
from priism.model.train import PRESETS  # noqa: E402


def tiny():
    torch.manual_seed(0)
    return AttractorSeparator(PRESETS["tiny"]).eval()


def test_bands_cover_all_bins():
    assert sum(default_bands(1025)) == 1025
    assert sum(PRESETS["tiny"].band_sizes()) == PRESETS["tiny"].n_fft // 2 + 1


def test_forward_shapes_and_exact_rest():
    model = tiny()
    mix = torch.randn(2, 2, 22050) * 0.1
    out = model(mix)
    k = PRESETS["tiny"].max_sources
    assert out["sources"].shape == (2, k, 2, 22050)
    assert out["exist_logits"].shape == (2, k)
    src, rest = model.separate(mix[0], threshold=0.0)
    torch.testing.assert_close(src.sum(0) + rest, mix[0], atol=1e-5, rtol=0)


def test_pit_loss_ignores_target_order():
    torch.manual_seed(1)
    est = torch.randn(1, 4, 2, 1000)
    tg = est[:, :3] + 0.01 * torch.randn(1, 3, 2, 1000)
    mix = tg.sum(1)
    logits = torch.zeros(1, 4)
    n = torch.tensor([3])
    a, sa = pit_loss(est, logits, tg, n, mix)
    b, sb = pit_loss(est, logits, tg[:, [2, 0, 1]], n, mix)
    assert abs(float(a) - float(b)) < 1e-4
    assert sa["sep_snr"] > 25  # each target found its near-copy (capped at 30 dB)


def test_neg_snr_soft_cap():
    x = torch.randn(2, 100)
    assert float(neg_snr(x, x)) == pytest.approx(-30.0, abs=0.01)


def test_dataset_merges_groups_and_drops_silence(tmp_path):
    from priism.gen.song import plan_song, write_song

    seed = next(s for s in range(3000) if any(x["split_notes"] for x in plan_song(s, 8)["sources"]))
    write_song(seed, tmp_path, duration_s=8, sample_rate=22050)
    ds = SongChunks(tmp_path, chunk_s=2.0, sample_rate=22050, items_per_song=4)
    groups = merged_sources(ds.metas[0])
    assert len(groups) == len(ds.metas[0]["sources"]) - 1  # the ambiguous pair is one target
    mix, tg = ds[0]
    assert mix.shape == (2, 44100) and tg.shape[1:] == (2, 44100) and tg.shape[0] <= len(groups)
    if tg.shape[0] == len(groups):
        torch.testing.assert_close(tg.sum(0), mix, atol=1e-4, rtol=0)
    m, t, n = collate([ds[0], ds[1]])
    assert m.shape[0] == 2 and t.shape[0] == 2 and int(n.max()) == t.shape[1]


def test_msst_attractor_separator_runs_on_a_small_roformer():
    import os
    import sys

    msst = os.environ.get("MSST_PATH")
    if not msst:
        pytest.skip("MSST_PATH not set (MSST checkout needed)")
    sys.path.insert(0, msst)
    pytest.importorskip("rotary_embedding_torch")
    from models.bs_roformer.bs_roformer import BSRoformer

    from priism.model.msst_core import MsstAttractorSeparator

    bands = (2,) * 24 + (4,) * 8 + (8,) * 4 + (17,)  # 129 bins for n_fft 256
    r = BSRoformer(dim=16, depth=1, stereo=True, num_stems=4, freqs_per_bands=bands, dim_head=8, heads=2,
                   stft_n_fft=256, stft_hop_length=64, stft_win_length=256, flash_attn=False)
    model = MsstAttractorSeparator(r, max_sources=5, decoder_depth=1, heads=2)
    mix = torch.randn(2, 2, 4096) * 0.1
    out = model(mix)
    assert out["sources"].shape == (2, 5, 2, 4096) and out["exist_logits"].shape == (2, 5)
    out["sources"].abs().mean().backward()
    with torch.autocast("cpu", dtype=torch.bfloat16):  # as on GPU (bf16): complex ops must stay in fp32
        assert model(mix)["sources"].dtype == torch.float32


def test_train_saves_and_resumes(tmp_path):
    from priism.gen.song import write_song
    from priism.model.train import train

    for s in range(2):
        write_song(s, tmp_path / "songs", duration_s=4, sample_rate=22050, n_sources=2)
    kw = dict(preset="tiny", batch=2, chunk_s=1.0, log_every=1, save_every=2, valid=tmp_path / "songs", log=lambda m: None)
    h = train(tmp_path / "songs", tmp_path / "run", steps=2, **kw)
    assert (tmp_path / "run" / "last.pt").exists() and "valid_sep_snr" in h[-1]
    logs = []
    h = train(tmp_path / "songs", tmp_path / "run", steps=4, **{**kw, "log": logs.append})
    assert any("resumed at step 3" in m for m in logs) and [x["step"] for x in h] == [1, 2, 3, 4]


def test_streamed_songs_feed_training(tmp_path):
    from priism.model.stream import LiveSongs, SongStream, ready_songs

    pool = SongStream(tmp_path / "pool", workers=2, duration_s=3, sample_rate=22050, max_songs=3, first_seed=5)
    try:
        pool.wait(min_songs=2, timeout_s=300)
    finally:
        pool.stop()
    assert 1 <= len(ready_songs(tmp_path / "pool")) <= 5
    it = iter(LiveSongs(tmp_path / "pool", chunk_s=1.0, sample_rate=22050))
    mix, tg = next(it)
    assert mix.shape == (2, 22050) and tg.shape[1:] == (2, 22050)


def test_group_outputs_sums_outputs_into_their_stem():
    from priism.model.evaluate import group_outputs

    torch.manual_seed(0)
    refs = torch.randn(3, 2, 500)
    est = torch.stack([0.6 * refs[0], 0.4 * refs[0], refs[2], refs[1]])
    torch.testing.assert_close(group_outputs(est, refs), refs)
