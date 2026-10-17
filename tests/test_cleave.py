"""Cleave phase 1: front end, masks, model shapes, and a CPU smoke run that must learn something."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf
import torch

from priism.lab.cleave.data import KSampler, find_songs, load_mix, parse_schedule
from priism.lab.cleave.features import MIDI_LO, N_PITCH, PL_DIM, SR, Frontend, mel_filterbank
from priism.lab.cleave.masks import sample_mask
from priism.lab.cleave.model import Cleave, CleaveConfig, patchify


def _tone(midi: float, seconds: float, harmonics: list[float], sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    return sum(a * np.sin(2 * np.pi * (h + 1) * f0 * t) for h, a in enumerate(harmonics)).astype(np.float32) * 0.2


def test_mel_filterbank_has_no_empty_band():
    fb = mel_filterbank()
    assert fb.shape[1] == 128
    assert np.allclose(fb.sum(0), 1.0)


def test_pitch_salience_finds_the_played_note():
    _, pl, level = Frontend()(torch.from_numpy(_tone(69, 1.0, [1, 0.5, 0.3, 0.2]))[None])
    assert pl.shape[-1] == PL_DIM
    frames = pl[0, 10:-10, :N_PITCH]
    assert (frames.argmax(-1) == 69 - MIDI_LO).float().mean() > 0.9
    assert ((frames > 0).sum(-1) <= 4).all()  # top-k cut
    assert level[0, 10:-10].min() > -30


def test_pitch_modes_name_the_chord_notes():
    """binary keeps salience positions at 1; cancel and bp should find a triad's three notes, not partials."""
    chord = sum(_tone(m, 1.5, [1, 0.6, 0.4, 0.3, 0.2]) for m in (57, 61, 64))  # A3 major
    wave = torch.from_numpy(chord)[None]
    _, sal, _ = Frontend("salience")(wave)
    _, binary, _ = Frontend("binary")(wave)
    assert torch.equal(binary[..., :N_PITCH] > 0, sal[..., :N_PITCH] > 0)
    assert set(binary[0, 20:-20, :N_PITCH].unique().tolist()) <= {0.0, 1.0}
    want = {57 - MIDI_LO, 61 - MIDI_LO, 64 - MIDI_LO}
    modes = ["cancel"]
    try:
        import onnxruntime  # noqa: F401

        from priism.lab.cleave.pitch_bp import model_path

        model_path()
        modes.append("bp")
    except (ImportError, FileNotFoundError):
        pass
    for mode in modes:
        _, pl, _ = Frontend(mode)(wave)
        found = (pl[0, 20:-20, :N_PITCH] > 0).float().mean(0) > 0.5
        assert want <= set(np.flatnonzero(found.numpy()).tolist()), mode


def test_masks_skip_silence_and_keep_something_visible():
    rng = np.random.default_rng(0)
    for _ in range(50):
        active = rng.random(20) > 0.3
        m = sample_mask(active, 12, 8, rng)
        assert not m[~active].any()
        if active.any():
            assert (~m[active].all(1)).any()


def test_patchify_order():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8)
    x = torch.arange(cfg.chunk_frames * cfg.n_mels, dtype=torch.float32).view(1, cfg.chunk_frames, cfg.n_mels)
    p = patchify(x, cfg)
    # patch (tcol 1, row 2) starts at frame 16, mel 32
    assert p[0, 1 * cfg.n_rows + 2, 0] == x[0, 16, 32]


def test_model_signals_have_one_value_per_song():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8)
    model = Cleave(cfg)
    B, C = 2, 3
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_patches, dtype=torch.bool)
    pmask[:, 1] = True
    out = model(patches, pl, active, pmask)
    assert torch.isfinite(out["loss"])
    for k in ("sigma", "pred_var", "pred_err"):
        assert out[k].shape == (B,)


def _songs(root, n_songs=4, seconds=16.0):
    rng = np.random.default_rng(0)
    timbres = [[1, 0.0, 0.0], [1, 0.8, 0.6, 0.4], [1, 0, 0.5, 0, 0.3]]
    for s in range(n_songs):
        d = root / f"song_{s}" / "sources"
        d.mkdir(parents=True)
        for i, h in enumerate(timbres):
            notes = [_tone(rng.integers(45, 70), 1.0, h) for _ in range(int(seconds))]
            sf.write(d / f"{i:02d}_timbre{i}.flac", np.concatenate(notes), SR)


def test_sampler_schedule():
    s = KSampler(10, parse_schedule("0:1,100:3"), mono_frac=0.0, seed=0)
    it = iter(s)
    assert all(next(it)[1] == 1 for _ in range(20))
    s.step = 100
    assert {next(it)[1] for _ in range(50)} == {2, 3}


def test_load_mix_uses_playing_stems(tmp_path):
    _songs(tmp_path, n_songs=1)
    song = find_songs([str(tmp_path)])[0]
    assert song.kinds == ["timbre0", "timbre1", "timbre2"]
    wave, used = load_mix(song, 2, np.random.default_rng(0), SR * 4)
    assert len(used) == 2 and wave.shape == (SR * 4,)


def test_cpu_smoke_run(tmp_path):
    from priism.lab.cleave.train import train

    _songs(tmp_path / "train")
    _songs(tmp_path / "valid", n_songs=2)
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8)
    summary = train([str(tmp_path / "train")], [str(tmp_path / "valid")], tmp_path / "run", cfg, steps=40,
                    batch=2, crop_chunks=4, warmup=5, k_schedule="0:1,20:2", eval_every=20, eval_k=2, per_k=4,
                    id_chunks=2, min_gap_s=2.0, workers=0, device="cpu", plateau=0)
    assert summary["step"] == 40
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert [h["step"] for h in hist] == [0, 20, 40]
    assert hist[-1]["loss"] < hist[1]["loss"] * 1.5
    assert "rho" in hist[-1]["sigma"]


def test_paired_views_and_collate():
    from priism.lab.cleave.data import FX_COLS, collate, paired_views

    rng = np.random.default_rng(0)
    x = _tone(57, 2.0, [1, 0.5, 0.3])
    a, b, fa, fb = paired_views(x, rng)
    assert a.shape == b.shape == x.shape and fa.shape == (len(FX_COLS),)
    assert np.abs(a).max() <= 0.99 + 1e-6 and not np.allclose(a, b)
    items = [{"wave": torch.from_numpy(a), "k": 1, "wave_b": torch.from_numpy(b), "fx_a": torch.from_numpy(fa),
              "fx_b": torch.from_numpy(fb)}, {"wave": torch.from_numpy(x), "k": 1}]
    out = collate(items)
    assert out["wave"].shape == (2, len(x)) and out["wave_b"].shape == (1, len(x))
    assert out["pair_idx"].tolist() == [0]
    assert collate(items[1:])["wave_b"].shape == (0, len(x))


def test_vbar_and_swap_losses():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, v_dim=4)
    model = Cleave(cfg)
    B, C = 3, 5
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_patches, dtype=torch.bool)
    pmask[:, 2] = True
    mu, _, _, v = model.encode(patches, active, pmask)
    assert v.shape == (B, C, 4)
    pi = torch.tensor([0, 2])
    out = model(patches, pl, active, pmask, pair=(pi, patches[pi] + 0.1 * torch.randn_like(patches[pi]), active[pi]))
    assert torch.isfinite(out["loss"]) and out["swap"] > 0 and out["inv"] >= 0
    out["loss"].backward()
    assert model.vbar.mlp[-1].weight.grad.abs().sum() > 0
    # V-bar of a chunk never reads that chunk
    patches2 = patches.clone()
    patches2[:, 0] += 5.0
    _, _, _, v2 = model.encode(patches2, active)
    _, _, _, v1 = model.encode(patches, active)
    assert torch.allclose(v1[:, 0], v2[:, 0], atol=1e-5) and not torch.allclose(v1[:, 1], v2[:, 1])


def test_no_v_checkpoint_still_loads():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8)
    state = {k: v for k, v in Cleave(cfg).state_dict().items()}
    d = cfg.to_dict()
    del d["v_dim"], d["v_window"]  # a phase 1a checkpoint predates them
    Cleave(CleaveConfig(**d)).load_state_dict(state)


def test_leakage_probes_on_toy_data():
    from priism.lab.cleave.train import logreg_bacc, ridge_r2

    rng = np.random.default_rng(0)
    n = 120
    groups = np.repeat(np.arange(n // 2), 2)
    Y = rng.random((n, 3))
    X_leak = np.concatenate([Y @ rng.normal(size=(3, 6)), rng.normal(size=(n, 4))], 1)
    X_clean = rng.normal(size=(n, 10))
    assert ridge_r2(X_leak, Y, groups) > 0.9
    assert ridge_r2(X_clean, Y, groups) < 0.1
    labels = np.repeat(rng.integers(0, 3, n // 2), 2)
    X_kind = np.eye(3)[labels] * 3 + rng.normal(size=(n, 3))
    bacc, chance = logreg_bacc(X_kind, labels, groups)
    assert chance == 1 / 3 and bacc > 0.8
    bacc_noise, _ = logreg_bacc(rng.normal(size=(n, 3)), labels, groups)
    assert bacc_noise < 0.6


def test_cpu_smoke_run_with_pairs(tmp_path):
    from priism.lab.cleave.train import train

    _songs(tmp_path / "train")
    _songs(tmp_path / "valid", n_songs=4)
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, v_dim=4)
    train([str(tmp_path / "train")], [str(tmp_path / "valid")], tmp_path / "run", cfg, steps=20, batch=2,
          crop_chunks=4, warmup=5, eval_every=20, eval_k=1, per_k=4, id_chunks=2, min_gap_s=2.0, workers=0,
          device="cpu", plateau=0, pair_frac=1.0, fx_tracks=12)
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert hist[-1]["swap"] > 0 and "fx_r2_z" in hist[-1] and "kind_bacc_v" in hist[-1]


def test_other_view_items(tmp_path):
    from priism.lab.cleave.data import MixDataset, collate

    _songs(tmp_path, n_songs=1)
    songs = find_songs([str(tmp_path)])
    n = SR * 2
    plain = MixDataset(songs, n)[(0, 1, 7)]
    paired = MixDataset(songs, n, pair_frac=1.0, other_view=True)[(0, 1, 7)]
    assert "wave_b" in paired and paired["wave_o"].shape == (n,)
    assert not torch.allclose(paired["wave_o"], paired["wave"])
    # the unpaired path draws the same crop as before the other view existed
    wave, _ = load_mix(songs[0], 1, np.random.default_rng(7), n)
    assert torch.allclose(plain["wave"], torch.from_numpy(wave))
    out = collate([paired, plain])
    assert out["wave_o"].shape == (1, n) and "wave_o" not in collate([plain])


def test_swap_reads_vbar_from_the_other_stem():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, v_dim=4)
    model = Cleave(cfg).eval()
    B, C = 2, 5
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_patches, dtype=torch.bool)
    pmask[:, 2] = True
    pi = torch.tensor([1])
    pb, po = patches[pi] + 0.1 * torch.randn_like(patches[pi]), torch.randn_like(patches[pi])
    o1 = model(patches, pl, active, pmask, pair=(pi, pb, active[pi], po, active[pi]), w_vinv=1.0)
    o2 = model(patches, pl, active, pmask, pair=(pi, pb, active[pi], po + 3.0, active[pi]), w_vinv=1.0)
    o3 = model(patches, pl, active, pmask, pair=(pi, pb, active[pi]))
    assert o1["vinv"] > 0 and o3["vinv"] == 0
    # InfoNCE invariance: a constant mu pays log(batch), the distance version would pay nothing
    model.glob.mu.weight.data.zero_()
    o4 = model(patches, pl, active, pmask, pair=(pi, pb, active[pi]))
    o5 = model(patches, pl, active, pmask, pair=(pi, pb, active[pi]), inv_mode="mse")
    assert torch.isclose(o4["inv"], torch.log(torch.tensor(float(B)))) and o5["inv"] < 1e-6
    model = Cleave(cfg).eval()
    assert not torch.isclose(o1["swap"], o2["swap"]) and torch.isclose(o1["pred"], o2["pred"])
    # dropping V-bar changes the main prediction only in training
    model.train()
    torch.manual_seed(0)
    d = model(patches, pl, active, pmask, v_drop=1.0)
    model.pred.v_embed.weight.data.zero_()  # V-bar then has no effect, as if dropped
    torch.manual_seed(0)
    z = model(patches, pl, active, pmask)
    assert torch.isclose(d["pred"], z["pred"])


def test_cpu_smoke_run_with_other_view(tmp_path):
    from priism.lab.cleave.train import train

    _songs(tmp_path / "train")
    _songs(tmp_path / "valid", n_songs=4)
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, v_dim=4)
    train([str(tmp_path / "train")], [str(tmp_path / "valid")], tmp_path / "run", cfg, steps=10, batch=2,
          crop_chunks=4, warmup=5, eval_every=10, eval_k=1, per_k=4, id_chunks=2, min_gap_s=2.0, workers=0,
          device="cpu", plateau=0, pair_frac=1.0, fx_tracks=0, other_view=True, v_drop=0.5, w_vinv=1.0)
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert hist[-1]["swap"] > 0 and hist[-1]["vinv"] > 0


def test_pool_mix_picks_crops_where_k_stems_play(tmp_path):
    from priism.lab.cleave.data import WIN, active_frac, load_song, pool_mix, pool_other

    d = tmp_path / "song" / "sources"
    d.mkdir(parents=True)
    tone = _tone(57, 12.0, [1, 0.5])
    late = np.concatenate([np.zeros(SR * 8, np.float32), tone[: SR * 4]])  # plays only in the last 4 s
    for i, x in enumerate([tone, late, np.zeros_like(tone)]):
        sf.write(d / f"{i:02d}_s{i}.flac", x, SR)
    loaded = load_song(find_songs([str(tmp_path)])[0])
    assert loaded.stems.dtype == np.float16 and loaded.active.shape == (3, 24)
    assert loaded.active[0].all() and not loaded.active[1, :16].any() and not loaded.active[2].any()
    rng = np.random.default_rng(0)
    for _ in range(20):
        wave, used, start = pool_mix(loaded, 2, rng, SR * 2)
        assert sorted(used) == [0, 1] and start % WIN == 0 and start >= SR * 8 - 3 * WIN  # >= 1 of 4 windows
        assert active_frac(loaded.stems[1, start:start + SR * 2].astype(np.float32)) >= 0.25
    _, used, start = pool_mix(loaded, 3, rng, SR * 2)  # no start has 3 playing stems: the most there are
    assert sorted(used) == [0, 1]
    assert pool_other(loaded, [1], start, SR * 2, rng) is not None
    assert pool_other(loaded, [0, 1], start, SR * 2, rng) is None


def test_pool_sampler_keeps_each_batch_in_its_pool():
    from priism.lab.cleave.data import PoolBatchSampler

    s = PoolBatchSampler(20, batch=4, schedule=parse_schedule("0:1"), pools=3, pool_songs=2, reuse=5, seed=0)
    it = iter(s)
    seen = set()
    for i in range(30):
        before = [list(p) for p in s.pools]
        items = next(it)
        assert len(items) == 4 and all(k == 1 for _, k, _ in items)
        assert {song for song, _, _ in items} <= set(before[i % 3]) | set(s.pools[i % 3])
        seen |= {song for song, _, _ in items}
    assert len(seen) > 6  # songs rotate out after `reuse` crops


def test_cpu_smoke_run_with_pool(tmp_path):
    from priism.lab.cleave.train import train

    _songs(tmp_path / "train")
    _songs(tmp_path / "valid", n_songs=2)
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8)
    summary = train([str(tmp_path / "train")], [str(tmp_path / "valid")], tmp_path / "run", cfg, steps=20,
                    batch=2, crop_chunks=4, warmup=5, k_schedule="0:1,10:2", eval_every=20, eval_k=2, per_k=4,
                    id_chunks=2, min_gap_s=2.0, workers=0, device="cpu", plateau=0, pool_gb=0.01, pool_reuse=4,
                    pair_frac=0.5, other_view=True)
    assert summary["step"] == 20 and summary["pool_songs"] >= 1


def test_context_branch_trains_the_patch_tokens():
    # docs/cleave.md 3.2: without it the local encoder's patch tokens get no prediction gradient
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, ctx_layers=1)
    model = Cleave(cfg)
    B, C = 2, 3
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_tcol, cfg.n_rows, dtype=torch.bool)
    pmask[:, 1] = True  # a section: no context inside the chunk
    pmask[:, 2, :, 2:4] = True  # a band: the context branch's target
    pmask = pmask.flatten(2)
    out = model(patches, pl, active, pmask, w_var=0.0, w_cov=0.0, w_ctx=1.0)
    assert float(out["ctx"]) > 0
    model.zero_grad()
    torch.manual_seed(0)
    with_ctx = model(patches, pl, active, pmask, w_var=0.0, w_cov=0.0, w_ctx=1.0)["loss"]
    torch.manual_seed(0)
    without = model(patches, pl, active, pmask, w_var=0.0, w_cov=0.0, w_ctx=0.0)["loss"]
    (with_ctx - without).backward()  # the context loss alone reaches the online local encoder, not the target
    assert model.local.embed.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.target.parameters())
    # older checkpoints (no ctx_layers) still load
    old = Cleave(CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8))
    assert old.ctx is None and float(old(patches, pl, active, pmask)["ctx"]) == 0


def test_cpu_smoke_run_with_context_branch(tmp_path):
    from priism.lab.cleave.train import train

    _songs(tmp_path / "train")
    _songs(tmp_path / "valid", n_songs=2)
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, ctx_layers=1)
    summary = train([str(tmp_path / "train")], [str(tmp_path / "valid")], tmp_path / "run", cfg, steps=20,
                    batch=2, crop_chunks=4, warmup=5, eval_every=20, eval_k=1, per_k=4, id_chunks=2,
                    min_gap_s=2.0, workers=0, device="cpu", plateau=0)
    assert summary["step"] == 20 and summary["w_ctx"] == 1.0
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert hist[-1]["ctx"] > 0


def test_mel_targets_are_the_normalised_patches():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, target="mel")
    model = Cleave(cfg)
    B, C = 2, 3
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_patches, dtype=torch.bool)
    pmask[:, 1] = True
    out = model(patches, pl, active, pmask)
    assert torch.isfinite(out["loss"]) and out["targets"].shape[-1] == cfg.patch_t * cfg.patch_f
    want = torch.nn.functional.layer_norm(patches[:, 1].reshape(-1, patches.shape[-1]), (patches.shape[-1],))
    assert torch.allclose(out["targets"].reshape(-1, patches.shape[-1]), want, atol=1e-5)


def test_zuse_hinge_reports_the_gap_and_trains_on_it():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, target="mel")
    model = Cleave(cfg)
    B, C = 3, 3
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_patches, dtype=torch.bool)
    pmask[:, 1] = True
    torch.manual_seed(0)
    a = model(patches, pl, active, pmask, beta=0.0, w_zuse=1.0, zuse_margin=10.0)
    torch.manual_seed(0)
    b = model(patches, pl, active, pmask, beta=0.0)
    assert float(b["zgap"]) == 0 and torch.isfinite(a["zgap"])
    assert abs(float(a["loss"] - b["loss"]) - (10.0 - float(a["zgap"]))) < 1e-3


def test_d2v_targets_average_normalised_layers():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=3, global_layers=1, pred_layers=1, z_dim=8, target="d2v",
                       d2v_k=2)
    model = Cleave(cfg)
    B, C = 2, 3
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    t = model.target.layer_targets(patches[0], 2)
    assert t.shape == (C, cfg.n_patches, cfg.dim)
    # the mean of two per-channel standardised layers: zero mean per channel and chunk
    assert t.mean(1).abs().max() < 1e-4
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_patches, dtype=torch.bool)
    pmask[:, 1] = True
    out = model(patches, pl, active, pmask, w_var=0.0, w_cov=0.0)
    assert torch.isfinite(out["loss"]) and out["targets"].shape[-1] == cfg.dim


def test_residual_mode_trains_a_pl_only_predictor_and_standardises_the_rest():
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, target="mel",
                       residual=True)
    model = Cleave(cfg)
    B, C = 2, 3
    patches = torch.randn(B, C, cfg.n_patches, cfg.patch_t * cfg.patch_f)
    pl = torch.rand(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
    active = torch.ones(B, C, dtype=torch.bool)
    pmask = torch.zeros(B, C, cfg.n_patches, dtype=torch.bool)
    pmask[:, 1] = True
    out = model(patches, pl, active, pmask)
    assert float(out["base"]) > 0 and torch.isfinite(out["loss"])
    t = out["targets"]
    assert abs(float(t.reshape(-1, t.shape[-1]).std(0).mean()) - 1) < 0.05  # standardised residual
    model.zero_grad()
    out["loss"].backward()
    assert model.base.out.weight.grad.abs().sum() > 0


def test_cpu_smoke_run_residual_reports_relative_z_use(tmp_path):
    from priism.lab.cleave.train import train

    _songs(tmp_path / "train")
    _songs(tmp_path / "valid", n_songs=2)
    cfg = CleaveConfig(dim=32, heads=2, local_layers=1, global_layers=1, pred_layers=1, z_dim=8, target="mel",
                       residual=True)
    train([str(tmp_path / "train")], [str(tmp_path / "valid")], tmp_path / "run", cfg, steps=20, batch=2,
          crop_chunks=4, warmup=5, eval_every=20, eval_k=1, per_k=4, id_chunks=2, min_gap_s=2.0, workers=0,
          device="cpu", plateau=0)
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert "z_use_rel" in hist[-1] and "pl_r2_z" in hist[-1] and hist[-1]["base"] > 0
