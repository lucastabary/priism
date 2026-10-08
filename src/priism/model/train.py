"""Training loop for the attractor separator (CPU-sized by default, same code on GPU)."""

from __future__ import annotations

import json
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from .data import SongChunks, collate
from .loss import count_accuracy, pit_loss
from .stream import LiveSongs, SongStream
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


def build_model(preset: str, msst: dict | None = None) -> tuple[torch.nn.Module, int, dict]:
    """Model, sample rate and the config saved next to its weights.

    ``preset="msst"`` wraps a pretrained MSST BS-RoFormer (``msst``: config, ckpt, path, max_sources).
    """
    if preset == "msst":
        from .msst_core import MsstAttractorSeparator, load_msst_roformer

        import yaml

        m = dict(msst or {})
        ckpt = m.get("ckpt")
        roformer = load_msst_roformer(m["config"], ckpt if ckpt not in (None, "None") else None, m["path"])
        model = MsstAttractorSeparator(roformer, max_sources=int(m.get("max_sources") or 16), grad_checkpoint=True)
        sr = yaml.load(Path(m["config"]).read_text(), Loader=yaml.FullLoader)["audio"]["sample_rate"]
        return model, sr, {"preset": preset, **{k: str(v) if isinstance(v, Path) else v for k, v in m.items()}}
    cfg = PRESETS[preset]
    return AttractorSeparator(cfg), cfg.sample_rate, {"preset": preset, **cfg.to_dict()}


def load_run(run: str | Path, device: str = "cpu", msst_path: str | Path | None = None) -> tuple[torch.nn.Module, int]:
    """Trained model (weights of ``run/model.pt``) and its sample rate. The MSST checkout can be moved."""
    cfg = json.loads((Path(run) / "config.json").read_text())
    preset = cfg.pop("preset")
    if preset == "msst":
        cfg = {**cfg, "ckpt": None, **({"path": msst_path} if msst_path else {})}
        model, sr, _ = build_model(preset, cfg)
    else:
        model, sr, _ = build_model(preset)
    model.load_state_dict(torch.load(Path(run) / "model.pt", map_location="cpu"))
    return model.to(device).eval(), sr


def train(data: str | Path | None, out: str | Path, preset: str = "tiny", steps: int = 200, batch: int = 4,
          chunk_s: float = 3.0, lr: float = 3e-4, device: str = "cpu", log_every: int = 10, workers: int = 0,
          seed: int = 0, lossy_p: float = 0.0, msst: dict | None = None, core_lr_scale: float = 0.1,
          valid: str | Path | None = None, valid_items: int = 64, save_every: int = 1000,
          stream: str | Path | None = None, stream_workers: int = 4, stream_songs: int = 400,
          stream_duration: float = 30.0, log=print) -> list[dict]:
    """Train; resumes from ``out/last.pt`` when it exists (pods get stopped).

    With ``stream`` (a local folder), training songs are generated during training by
    ``stream_workers`` processes (rolling pool of ``stream_songs``) instead of read from ``data``.

    With a pretrained core (``preset="msst"``), the core learns at ``lr * core_lr_scale`` so the new
    attractor parts move fast without wrecking what the core knows.
    """
    torch.manual_seed(seed)
    model, sr, saved_cfg = build_model(preset, msst)
    model = model.to(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(saved_cfg, indent=1))
    log(f"{n_params(model) / 1e6:.2f} M parameters")

    pool = None
    if stream:
        pool = SongStream(stream, stream_workers, stream_duration, sr, max_songs=stream_songs)
        pool.wait(min_songs=max(8, 2 * batch), log=log)
        ds = LiveSongs(stream, chunk_s, sr, seed=seed, lossy_p=lossy_p)
        dl = DataLoader(ds, batch_size=batch, collate_fn=collate, num_workers=workers, persistent_workers=workers > 0)
    else:
        ds = SongChunks(data, chunk_s, sr, items_per_song=10**6, seed=seed, lossy_p=lossy_p)
        # Sampling with replacement: a shuffle of len(ds) indices (songs x 10**6 crops) would not fit in memory.
        sampler = RandomSampler(ds, replacement=True, num_samples=steps * batch)
        dl = DataLoader(ds, batch_size=batch, sampler=sampler, collate_fn=collate, num_workers=workers,
                        drop_last=True, persistent_workers=workers > 0)
    core = [p for n, p in model.named_parameters() if n.startswith("r.")]
    new = [p for n, p in model.named_parameters() if not n.startswith("r.")]
    groups = [{"params": new, "lr": lr}] + ([{"params": core, "lr": lr * core_lr_scale}] if core else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.05)
    history, start = [], 1
    last = out / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        history, start = st["history"], st["step"] + 1
        with warnings.catch_warnings():  # replay the schedule: also right when --steps changed since the save
            warnings.simplefilter("ignore")
            for _ in range(start - 1):
                sched.step()
        log(f"resumed at step {start}")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    vds = SongChunks(valid, chunk_s, sr, items_per_song=1, seed=1) if valid else None
    try:
        _loop(model, dl, opt, sched, vds, valid_items, amp, device, start, steps, history, out, last, log_every,
              save_every, log)
    finally:
        if pool:
            pool.stop()
    return history


def _loop(model, dl, opt, sched, vds, valid_items, amp, device, start, steps, history, out, last, log_every,
          save_every, log):
    t0 = time.time()
    it = iter(dl)
    model.train()
    for step in range(start, steps + 1):
        try:
            mix, tg, n = next(it)
        except StopIteration:
            it = iter(dl)
            mix, tg, n = next(it)
        mix, tg = mix.to(device), tg.to(device)
        with amp:
            o = model(mix)
        loss, stats = pit_loss(o["sources"].float(), o["exist_logits"].float(), tg, n, mix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        stats.update(step=step, count_acc=count_accuracy(o["exist_logits"].detach(), n), sec=round(time.time() - t0, 1))
        if step % log_every == 0 or step == steps:
            log(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
        if (step % save_every == 0 or step == steps) and vds is not None:
            stats.update({f"valid_{k}": v for k, v in evaluate(model, vds, valid_items, device, amp).items()})
            log(" ".join(f"{k}={v:.3g}" for k, v in stats.items() if k.startswith("valid_")))
        history.append(stats)
        if step % save_every == 0 or step == steps:
            torch.save(model.state_dict(), out / "model.pt")  # weights only, for inference and copies
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "history": history}, out / "last.tmp")
            (out / "last.tmp").replace(last)
            (out / "history.json").write_text(json.dumps(history))


@torch.no_grad()
def evaluate(model: torch.nn.Module, ds: SongChunks, items: int, device: str, amp=None) -> dict:
    """Average PIT stats over the first ``items`` crops of a fixed set (one crop per song)."""
    model.eval()
    acc: dict[str, float] = {}
    k = min(items, len(ds))
    for i in range(k):
        mix, tg, n = collate([ds[i]])
        mix, tg = mix.to(device), tg.to(device)
        with amp or nullcontext():
            o = model(mix)
        _, st = pit_loss(o["sources"].float(), o["exist_logits"].float(), tg, n, mix)
        st["count_acc"] = count_accuracy(o["exist_logits"], n)
        for key, v in st.items():
            acc[key] = acc.get(key, 0.0) + v / k
    model.train()
    return acc
