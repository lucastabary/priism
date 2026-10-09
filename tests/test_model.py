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
    from priism.model.distill import teacher_stems

    with torch.autocast("cpu", dtype=torch.bfloat16):  # the stock model as teacher, inside the bf16 training step
        teacher = BSRoformer(dim=16, depth=1, stereo=True, num_stems=4, freqs_per_bands=bands, dim_head=8, heads=2,
                             stft_n_fft=256, stft_hop_length=64, stft_win_length=256, flash_attn=False)
        assert teacher_stems(teacher, torch.randn(3, 2, 4096) * 0.1).shape == (3, 4, 2, 4096)


def test_train_saves_and_resumes(tmp_path):
    from priism.gen.song import write_song
    from priism.model.train import train

    for s in range(2):
        write_song(s, tmp_path / "songs", duration_s=4, sample_rate=22050, n_sources=2)
    (tmp_path / "twins").symlink_to(tmp_path / "songs")
    kw = dict(preset="tiny", batch=2, chunk_s=1.0, log_every=1, save_every=2, valid=[tmp_path / "songs", tmp_path / "twins"],
              log=lambda m: None)
    h = train(tmp_path / "songs", tmp_path / "run", steps=2, **kw)
    assert (tmp_path / "run" / "last.pt").exists() and "valid_sep_snr" in h[-1] and "valid_twins_sep_snr" in h[-1]
    logs = []
    h = train(tmp_path / "songs", tmp_path / "run", steps=4, **{**kw, "log": logs.append})
    assert any("resumed at step 3" in m for m in logs) and [x["step"] for x in h] == [0, 1, 2, 3, 4]
    assert "valid_twins_sep_snr" in h[0]  # starting point, to judge the run against


def test_probe_stops_early_then_resumes(tmp_path):
    from priism.gen.song import write_song
    from priism.model.train import train

    write_song(0, tmp_path / "songs", duration_s=4, sample_rate=22050, n_sources=2)
    kw = dict(preset="tiny", batch=2, chunk_s=1.0, log_every=1, save_every=10, valid=tmp_path / "songs",
              log=lambda m: None)
    h = train(tmp_path / "songs", tmp_path / "run", steps=6, stop_at=2, **kw)
    assert [x["step"] for x in h] == [0, 1, 2] and "valid_sep_snr" in h[-1]
    h = train(tmp_path / "songs", tmp_path / "run", steps=6, **kw)
    assert [x["step"] for x in h] == list(range(7))


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


def test_stream_curriculum_limits_sources(tmp_path):
    import json

    from priism.model.stream import SongStream, ready_songs

    pool = SongStream(tmp_path / "pool", workers=2, duration_s=2, sample_rate=22050, max_songs=10, first_seed=7,
                      sources=(2, 3))
    try:
        pool.wait(min_songs=3, timeout_s=300)
    finally:
        pool.stop()
    for song in ready_songs(tmp_path / "pool"):
        assert 2 <= len(json.loads((song / "meta.json").read_text())["sources"]) <= 3


def test_group_loss_rewards_outputs_that_sum_to_the_stems():
    from priism.model.distill import group_loss

    torch.manual_seed(0)
    refs = torch.randn(1, 3, 2, 800)
    good = torch.stack([0.5 * refs[0, 0], 0.5 * refs[0, 0], refs[0, 1], refs[0, 2]])[None]
    bad = torch.randn(1, 4, 2, 800)
    assert group_loss(good, refs)[1] > 25 > group_loss(bad, refs)[1]


def test_msst_training_with_real_song_distillation(tmp_path):
    import os

    import numpy as np
    import soundfile as sf

    from priism.gen.song import write_song
    from priism.model.train import train

    msst = os.environ.get("MSST_PATH")
    if not msst:
        pytest.skip("MSST_PATH not set (MSST checkout needed)")
    pytest.importorskip("rotary_embedding_torch")
    bands = ", ".join(["2"] * 24 + ["4"] * 8 + ["8"] * 4 + ["17"])  # 129 bins for n_fft 256
    cfg = tmp_path / "small.yaml"
    cfg.write_text(f"""audio:
  sample_rate: 22050
model:
  dim: 16
  depth: 1
  stereo: true
  num_stems: 4
  freqs_per_bands: !!python/tuple [{bands}]
  dim_head: 8
  heads: 2
  stft_n_fft: 256
  stft_hop_length: 64
  stft_win_length: 256
  flash_attn: false
""")
    write_song(1, tmp_path / "songs", duration_s=3, sample_rate=22050, n_sources=2)
    (tmp_path / "real").mkdir()
    sf.write(tmp_path / "real" / "a.flac", 0.1 * np.random.randn(22050 * 3, 2), 22050)
    logs = []
    h = train(tmp_path / "songs", tmp_path / "run", preset="msst", steps=2, batch=1, chunk_s=0.5, log_every=1,
              msst={"config": cfg, "ckpt": None, "path": msst, "max_sources": 4}, real=tmp_path / "real",
              real_every=1, log=logs.append)
    assert "real_snr" in h[-1] and any("distillation on 1 real songs" in m for m in logs)


def test_plateau_detection():
    from priism.model.train import _stalled

    h = [{"step": 0, "valid_sep_snr": 5.0}, {"step": 1, "valid_sep_snr": 8.0}, {"step": 2, "valid_sep_snr": 8.05},
         {"step": 3, "valid_sep_snr": 7.9}]
    assert _stalled(h, 2, 0.1) and not _stalled(h, 3, 0.1)
    h.append({"step": 4, "valid_sep_snr": 7.0, "valid_twins_sep_snr": 1.0})  # a set still gaining keeps it going
    h[0]["valid_twins_sep_snr"] = 0.5
    assert not _stalled(h, 2, 0.1)


def test_grouping_ignores_near_silent_stems():
    from priism.model.distill import group_loss
    from priism.model.evaluate import group_outputs

    torch.manual_seed(0)
    refs = torch.randn(1, 4, 2, 2000)
    refs[0, 2:] *= 1e-4  # stems the teacher found (almost) nothing for
    est = torch.cat([refs[:, :2] * 0.5, refs[:, :2] * 0.5, torch.zeros(1, 4, 2, 2000)], 1)
    assert group_loss(est, refs)[1] > 25
    g = group_outputs(est[0], refs[0])
    torch.testing.assert_close(g[:2], refs[0, :2])


def test_validation_scores_twins_alone(tmp_path, monkeypatch):
    from priism.gen import song
    from priism.model.data import SongChunks
    from priism.model.train import build_model, evaluate

    monkeypatch.setattr(song, "TWIN_P", 1.0)
    monkeypatch.setattr(song, "TWIN_EXTRA_P", 1.0)
    seed = next(s for s in range(200) if sum(x["twin_of"] is not None and x["merge_group"] is None
                                             for x in song.plan_song(s, 6.0, n_sources=5)["sources"]) >= 2)
    song.write_song(seed, tmp_path / "twins", duration_s=6, sample_rate=22050, n_sources=5)
    ds = SongChunks(tmp_path / "twins", 4.0, 22050, items_per_song=1, seed=1)
    mix, tg, fam = ds.item_with_families(0)
    assert len(fam) == tg.shape[0]
    model, _, _ = build_model("tiny")
    st = evaluate(model, ds, 1, "cpu")
    assert max(fam.count(f) for f in fam) > 1
    assert {"twin_snr", "twin_all5", "other_snr"} <= set(st)


def test_msst_v2_starts_from_v1_weights_and_runs(tmp_path):
    import os
    import sys

    msst = os.environ.get("MSST_PATH")
    if not msst:
        pytest.skip("MSST_PATH not set (MSST checkout needed)")
    sys.path.insert(0, msst)
    pytest.importorskip("rotary_embedding_torch")
    from models.bs_roformer.bs_roformer import BSRoformer

    from priism.model.msst_core import MsstAttractorSeparator

    bands = (2,) * 24 + (4,) * 8 + (8,) * 4 + (17,)
    mk = lambda: BSRoformer(dim=16, depth=1, stereo=True, num_stems=4, freqs_per_bands=bands, dim_head=8, heads=2,
                            stft_n_fft=256, stft_hop_length=64, stft_win_length=256, flash_attn=False)
    torch.manual_seed(0)
    v1 = MsstAttractorSeparator(mk(), max_sources=5, decoder_depth=1, heads=2)
    v2 = MsstAttractorSeparator(mk(), max_sources=5, decoder_depth=1, heads=2, v2=True)
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith(("grid.", "refiner.")) for k in missing)
    mix = torch.randn(2, 2, 4096) * 0.1
    xk = torch.randn(3, 9, len(bands), 16)
    torch.testing.assert_close(v2.refiner(xk), xk)  # identity at step 0
    out = v2(mix)
    assert out["sources"].shape == (2, 5, 2, 4096)
    out["sources"].abs().mean().backward()
    assert v2.refiner.out.weight.grad is not None and v2.grid.band.grad is not None


def test_slot_attention_queries_are_exchangeable_and_deterministic_in_eval(tmp_path):
    import os
    import sys

    msst = os.environ.get("MSST_PATH")
    if not msst:
        pytest.skip("MSST_PATH not set (MSST checkout needed)")
    sys.path.insert(0, msst)
    pytest.importorskip("rotary_embedding_torch")
    from models.bs_roformer.bs_roformer import BSRoformer

    from priism.model.msst_core import MsstAttractorSeparator

    bands = (2,) * 24 + (4,) * 8 + (8,) * 4 + (17,)
    mk = lambda: BSRoformer(dim=16, depth=1, stereo=True, num_stems=4, freqs_per_bands=bands, dim_head=8, heads=2,
                            stft_n_fft=256, stft_hop_length=64, stft_win_length=256, flash_attn=False)
    v1 = MsstAttractorSeparator(mk(), max_sources=5, decoder_depth=1, heads=2)
    m = MsstAttractorSeparator(mk(), max_sources=5, decoder_depth=1, heads=2, slot_attention=True)
    missing, unexpected = m.load_state_dict(v1.state_dict(), strict=False)
    assert not unexpected and any(k.startswith("slots.") for k in missing)
    mix = torch.randn(2, 2, 4096) * 0.1
    out = m(mix)
    assert out["sources"].shape == (2, 5, 2, 4096)
    out["sources"].abs().mean().backward()
    assert m.slots.to_q.weight.grad is not None
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(mix)["sources"], m(mix)["sources"])  # fixed draws at inference
