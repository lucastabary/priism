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
