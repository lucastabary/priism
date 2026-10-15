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
