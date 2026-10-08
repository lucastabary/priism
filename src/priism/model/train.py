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
        model = MsstAttractorSeparator(roformer, max_sources=int(m.get("max_sources") or 16), grad_checkpoint=True,
                                       v2=bool(m.get("v2")))
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
          valid: str | Path | list | None = None, valid_items: int = 64, save_every: int = 1000,
          exist_weight: float = 1.0,
          stream: str | Path | None = None, stream_workers: int = 4, stream_songs: int = 400,
          stream_duration: float = 30.0, stream_sources: tuple[int, int] | None = None,
          init: str | Path | None = None, real: str | Path | None = None, real_every: int = 2,
          real_weight: float = 1.0, stop_at: int | None = None, plateau: int = 0, plateau_delta: float = 0.1,
          log=print) -> list[dict]:
    """Train; resumes from ``out/last.pt`` when it exists (pods get stopped).

    With ``stream`` (a local folder), training songs are generated during training by
    ``stream_workers`` processes (rolling pool of ``stream_songs``) instead of read from ``data``.
    ``init`` starts from the weights of an earlier run (``model.pt``), e.g. the previous curriculum stage.
    ``real`` (a folder of real songs, msst preset only) adds, every ``real_every`` steps, a distillation
    batch: the frozen pretrained model's stems guide our outputs grouped per stem (see distill.py).
    ``stop_at`` ends after that step (saved) with the schedule of ``steps``: a short probe that a later
    call with the same ``out`` resumes into the full run. A fresh run scores the validation sets at step 0
    first, so a probe can be judged against its own starting point.
    ``plateau`` > 0 ends the run (saved) once ``plateau`` validations in a row have improved none of the
    validation sets' separation by ``plateau_delta`` dB over its best: a stalled run frees the GPU.

    With a pretrained core (``preset="msst"``), the core learns at ``lr * core_lr_scale`` so the new
    attractor parts move fast without wrecking what the core knows.
    """
    torch.manual_seed(seed)
    model, sr, saved_cfg = build_model(preset, msst)
    if init:
        state = torch.load(init, map_location="cpu")
        if preset == "msst" and (msst or {}).get("v2"):  # a v1 run's weights: the v2 parts start at identity
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or any(not k.startswith(("grid.", "refiner.")) for k in missing):
                raise ValueError(f"{init} does not fit: missing {missing}, unexpected {unexpected}")
            if missing:
                log(f"new v2 parts start fresh: {len(missing)} tensors")
        else:
            model.load_state_dict(state)
        log(f"weights from {init}")
    model = model.to(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(saved_cfg, indent=1))
    log(f"{n_params(model) / 1e6:.2f} M parameters")

    pool = None
    if stream:
        pool = SongStream(stream, stream_workers, stream_duration, sr, max_songs=stream_songs, sources=stream_sources)
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
    # Several validation sets: the first logs as valid_*, the others as valid_<folder name>_*.
    valid = [valid] if isinstance(valid, (str, Path)) else list(valid or [])
    vds = [(("" if i == 0 else Path(v).name + "_"), SongChunks(v, chunk_s, sr, items_per_song=1, seed=1))
           for i, v in enumerate(valid)]
    teach = None
    if real:
        from .distill import RealCrops, load_teacher

        if preset != "msst":
            raise ValueError("real-song distillation needs the msst preset (its pretrained model is the teacher)")
        rdl = DataLoader(RealCrops(real, chunk_s, sr, seed=seed), batch_size=batch, num_workers=2,
                         persistent_workers=True)
        teach = {"it": iter(rdl), "model": load_teacher(msst["config"], msst["ckpt"], msst["path"], device),
                 "every": real_every, "weight": real_weight}
        log(f"distillation on {len(rdl.dataset.files)} real songs, every {real_every} steps")
    try:
        _loop(model, dl, opt, sched, vds, valid_items, amp, device, start, steps, history, out, last, log_every,
              save_every, log, exist_weight, teach, stop_at, plateau, plateau_delta)
    finally:
        if pool:
            pool.stop()
    return history


def _loop(model, dl, opt, sched, vds, valid_items, amp, device, start, steps, history, out, last, log_every,
          save_every, log, exist_weight=1.0, teach=None, stop_at=None, plateau=0, plateau_delta=0.1):
    t0 = time.time()
    if start == 1 and vds:  # the starting point, to judge what the run brings
        base = {"step": 0}
        for name, ds in vds:
            base.update({f"valid_{name}{k}": v for k, v in evaluate(model, ds, valid_items, device, amp).items()})
        log(" ".join(f"{k}={v:.3g}" for k, v in base.items()))
        history.append(base)
    end = min(steps, stop_at) if stop_at else steps
    it = iter(dl)
    model.train()
    for step in range(start, end + 1):
        try:
            mix, tg, n = next(it)
        except StopIteration:
            it = iter(dl)
            mix, tg, n = next(it)
        mix, tg = mix.to(device), tg.to(device)
        with amp:
            o = model(mix)
        loss, stats = pit_loss(o["sources"].float(), o["exist_logits"].float(), tg, n, mix,
                               exist_weight=exist_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if teach and step % teach["every"] == 0:
            # Separate backward: both graphs at once would not fit in 24 GB.
            from .distill import group_loss, teacher_stems

            real_mix = next(teach["it"]).to(device)
            refs = teacher_stems(teach["model"], real_mix)  # (B, stems, C, S)
            with amp:
                ro = model(real_mix)
            rloss, stats["real_snr"] = group_loss(ro["sources"].float(), refs)
            (teach["weight"] * rloss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        stats.update(step=step, count_acc=count_accuracy(o["exist_logits"].detach(), n), sec=round(time.time() - t0, 1))
        if step % log_every == 0 or step == steps:
            log(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
        if (step % save_every == 0 or step == end) and vds:
            for name, ds in vds:
                stats.update({f"valid_{name}{k}": v for k, v in evaluate(model, ds, valid_items, device, amp).items()})
            log(" ".join(f"{k}={v:.3g}" for k, v in stats.items() if k.startswith("valid_")))
        history.append(stats)
        stalled = plateau and vds and step % save_every == 0 and _stalled(history, plateau, plateau_delta)
        if stalled:
            log(f"plateau: no validation set gained {plateau_delta} dB in {plateau} validations, stopping at step {step}")
        if step % save_every == 0 or step == end or stalled:
            torch.save(model.state_dict(), out / "model.pt")  # weights only, for inference and copies
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "history": history}, out / "last.tmp")
            (out / "last.tmp").replace(last)
            (out / "history.json").write_text(json.dumps(history))
        if stalled:
            (out / "PLATEAU").write_text(str(step))
            break


def _stalled(history: list[dict], patience: int, delta: float) -> bool:
    """True when the last ``patience`` validations beat none of the earlier bests by ``delta`` dB."""
    vals = [x for x in history if any(k.startswith("valid_") and k.endswith("sep_snr") for k in x)]
    if len(vals) <= patience:
        return False
    keys = [k for k in vals[-1] if k.startswith("valid_") and k.endswith("sep_snr")]
    old, new = vals[:-patience], vals[-patience:]
    return not any(max(x.get(k, -1e9) for x in new) >= max(x.get(k, -1e9) for x in old) + delta for k in keys)


class _WithFamilies(torch.utils.data.Dataset):
    def __init__(self, ds: SongChunks):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.ds.item_with_families(i)


def kept_scores(sources: torch.Tensor, exist_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """SNR (dB) of each target as ``split`` would deliver it: only outputs kept by the existence head
    (p > 0.5), matched one to one; a target left without output scores as silence (0 dB)."""
    from scipy.optimize import linear_sum_assignment

    from .loss import neg_snr

    est = sources[exist_logits.sigmoid() > 0.5]
    snr = torch.zeros(len(targets))
    if len(est) and len(targets):
        cost = neg_snr(est[:, None], targets[None])
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        snr[torch.as_tensor(cols)] = -cost[rows, cols].cpu()
    return snr


@torch.no_grad()
def evaluate(model: torch.nn.Module, ds: SongChunks, items: int, device: str, amp=None) -> dict:
    """Average PIT stats over the first ``items`` crops of a fixed set (one crop per song).

    On sets holding twins (one instrument playing several parts), also scores the twins alone the way
    ``split`` delivers them (kept outputs only): ``twin_snr`` (mean over twin parts), ``twin_all5`` (share of
    twin families with every heard part above 5 dB) and ``other_snr`` (the other sources, same rule).
    The mean over all sources (``sep_snr``) hides twins that stay merged.
    """
    from collections import Counter

    model.eval()
    acc: dict[str, float] = {}
    twin, other, fam_ok, fams = [], [], 0, 0
    k = min(items, len(ds))
    # Crops are read by worker processes ahead of the model: reading every stem from the network volume
    # one item at a time left the GPU idle for minutes per validation.
    loader = DataLoader(_WithFamilies(ds), batch_size=None, sampler=range(k), num_workers=min(4, k),
                        collate_fn=lambda x: x)
    for m1, t1, families in loader:
        mix, tg, n = collate([(m1, t1)])
        mix, tg = mix.to(device), tg.to(device)
        with amp or nullcontext():
            o = model(mix)
        src, ex = o["sources"].float(), o["exist_logits"].float()
        _, st = pit_loss(src, ex, tg, n, mix)
        st["count_acc"] = count_accuracy(o["exist_logits"], n)
        for key, v in st.items():
            acc[key] = acc.get(key, 0.0) + v / k
        size = Counter(families)
        if any(c > 1 for c in size.values()):
            snr = kept_scores(src[0], ex[0], tg[0, :int(n[0])])
            for f in {f for f in families if size[f] > 1}:
                parts = [float(s) for s, g in zip(snr, families) if g == f]
                twin += parts
                fam_ok += all(s > 5 for s in parts)
                fams += 1
            other += [float(s) for s, g in zip(snr, families) if size[g] == 1]
    if twin:
        acc.update(twin_snr=sum(twin) / len(twin), twin_all5=fam_ok / fams,
                   other_snr=sum(other) / max(len(other), 1))
    model.train()
    return acc
