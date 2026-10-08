"""Training loop for the attractor separator (CPU-sized by default, same code on GPU)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SongChunks, collate
from .loss import count_accuracy, pit_loss
from .separator import AttractorSeparator, SeparatorConfig, n_params

PRESETS = {
    # Fits a laptop CPU: for checking that the whole chain learns, not for quality.
    "tiny": SeparatorConfig(sample_rate=22050, n_fft=512, hop=128, dim=32, depth=1, heads=2, max_sources=8,
                            decoder_depth=1, head_hidden=64),
    "small": SeparatorConfig(sample_rate=44100, n_fft=2048, hop=512, dim=96, depth=3, heads=4, max_sources=16,
                             decoder_depth=2, head_hidden=192),
    "base": SeparatorConfig(sample_rate=44100, n_fft=2048, hop=441, dim=256, depth=8, heads=8, max_sources=16,
                            decoder_depth=3, head_hidden=1024),
}


def train(data: str | Path, out: str | Path, preset: str = "tiny", steps: int = 200, batch: int = 4,
          chunk_s: float = 3.0, lr: float = 3e-4, device: str = "cpu", log_every: int = 10, workers: int = 0,
          seed: int = 0, lossy_p: float = 0.0, log=print) -> list[dict]:
    torch.manual_seed(seed)
    cfg = PRESETS[preset]
    model = AttractorSeparator(cfg).to(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"preset": preset, **cfg.to_dict()}, indent=1))
    log(f"{n_params(model) / 1e6:.2f} M parameters")

    ds = SongChunks(data, chunk_s, cfg.sample_rate, items_per_song=10**6, seed=seed, lossy_p=lossy_p)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    history = []
    t0 = time.time()
    it = iter(dl)
    for step in range(1, steps + 1):
        mix, tg, n = next(it)
        mix, tg = mix.to(device), tg.to(device)
        o = model(mix)
        loss, stats = pit_loss(o["sources"], o["exist_logits"], tg, n, mix)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        stats.update(step=step, count_acc=count_accuracy(o["exist_logits"].detach(), n), sec=round(time.time() - t0, 1))
        history.append(stats)
        if step % log_every == 0 or step == steps:
            log(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
    torch.save(model.state_dict(), out / "model.pt")  # weights only: small files (shared folder drops huge ones)
    (out / "history.json").write_text(json.dumps(history))
    return history
