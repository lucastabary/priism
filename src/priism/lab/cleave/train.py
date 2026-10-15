"""Training loop of the Cleave JEPA, with the lab's guards and the phase 1 measurements built in.

Every evaluation appends one JSON line to ``history.jsonl`` (flushed) and prints it:

- ``id_auc`` / ``id_auc_kind``: Z (posterior mean) of two crops of the same track, ``min_gap_s`` apart, against
  other tracks (all, or of the same kind): the identity criterion C2 of docs/cleave.md. Step 0 is the
  randomly initialised model, the baseline C2 has to beat.
- ``z_rank`` / ``target_rank``: effective ranks of Z and of the target patch tokens (criterion C1).
- ``z_use``: validation prediction loss with Z shuffled across songs minus with the true Z (C1: the
  predictor must need Z, else P and L leak the timbre).
- per k in 1..``eval_k``: the three uncertainty signals (posterior sigma, predictor variance, predictor
  error) and their Spearman correlation with k over the validation mixes (docs/cleave.md 2.8).

Guards: non-finite loss, Z collapse (effective rank below ``min_z_rank`` after ``warmup``), a plateau of
``id_auc_kind`` (or ``id_auc`` when kinds are unknown), ``max_minutes``. ``--measure`` only times steps
(data wait vs compute, peak GPU memory) and extrapolates, the step before any long run.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from .data import KSampler, MixDataset, Song, active_frac, find_songs, load_mix, parse_schedule, read_crop, song_frames_sr
from .features import HOP, SILENCE_DB, SR, Frontend
from .masks import sample_mask
from .model import Cleave, CleaveConfig, effective_rank


def _log(*a) -> None:  # flushed: a redirected log shows progress as it happens
    print(*a, flush=True)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Batcher:
    """Waveforms -> model inputs on the device: patches, P/L, active chunks, and (if asked) sampled masks."""

    def __init__(self, cfg: CleaveConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.front = Frontend().to(device)

    def __call__(self, wave: torch.Tensor, model: Cleave, rng: np.random.Generator | None):
        logmel, pl, level = self.front(wave.to(self.device))
        patches, plc = model.chunk(logmel, pl)
        B, C = patches.shape[:2]
        active = level[:, : C * self.cfg.chunk_frames].reshape(B, C, -1).amax(-1) > SILENCE_DB
        pmask = None
        if rng is not None:
            act = active.cpu().numpy()
            pmask = torch.from_numpy(np.stack([sample_mask(a, self.cfg.n_tcol, self.cfg.n_rows, rng) for a in act]))
            pmask = pmask.to(self.device)
        return patches, plc, active, pmask


def id_pairs(songs: list[Song], crop: int, gap: int, max_tracks: int, seed: int = 0) -> list[tuple]:
    """(song, stem, start A, start B, kind) for tracks long enough to give two playing crops ``gap`` apart."""
    rng = np.random.default_rng(seed)
    out = []
    for si in rng.permutation(len(songs)):
        song = songs[si]
        total = song_frames_sr(song)
        if total < 2 * crop + gap:
            continue
        for st in range(len(song.stems)):
            a = int(rng.integers(0, total - 2 * crop - gap + 1))
            b = int(rng.integers(a + crop + gap, total - crop + 1))
            if min(active_frac(read_crop(song, st, a, crop)), active_frac(read_crop(song, st, b, crop))) < 0.25:
                continue
            out.append((int(si), st, a, b, song.kinds[st]))
            if len(out) >= max_tracks:
                return out
    return out


def k_sets(songs: list[Song], n: int, eval_k: int, per_k: int, seed: int = 0) -> list[tuple[np.ndarray, int]]:
    """Fixed validation mixes: for each k in 1..eval_k, ``per_k`` crops with exactly k playing stems."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(1, eval_k + 1):
        got, tries = 0, 0
        while got < per_k and tries < 20 * per_k:
            tries += 1
            wave, used = load_mix(songs[int(rng.integers(len(songs)))], k, rng, n)
            if len(used) == k:
                out.append((wave, k))
                got += 1
    return out


@torch.no_grad()
def evaluate(model: Cleave, batcher: Batcher, val_songs: list[Song], pairs: list[tuple], ksets: list, crop: int,
             batch: int, seed: int = 0) -> dict:
    model.eval()
    rec: dict = {}
    # identity: Z of crop A vs crop B of each track
    zs = []
    for i in range(0, len(pairs), batch):
        part = pairs[i:i + batch]
        waves = [read_crop(val_songs[s], st, x, crop) for s, st, a, b, _ in part for x in (a, b)]
        patches, _, active, _ = batcher(torch.from_numpy(np.stack(waves)), model, None)
        mu, _, _ = model.encode(patches, active)
        zs.append(torch.nn.functional.normalize(mu.float(), dim=-1).cpu())
    if zs:
        z = torch.cat(zs).view(-1, 2, zs[0].shape[-1])
        za, zb = z[:, 0], z[:, 1]
        sim = (za @ zb.T).numpy()
        kinds = np.array([p[4] for p in pairs])
        eye = np.eye(len(pairs), dtype=bool)
        same_kind = kinds[:, None] == kinds[None, :]
        rec["id_auc"] = _auc(sim[eye], sim[~eye])
        rec["id_auc_kind"] = _auc(sim[eye], sim[~eye & same_kind])
        rec["z_rank"] = effective_rank(torch.cat([za, zb]))
        rec["n_tracks"] = len(pairs)
    # uncertainty signals per k, prediction loss, Z usage
    rng = np.random.default_rng(seed)
    sig = {"sigma": [], "pred_var": [], "pred_err": []}
    ks, pred, pred_shuf, targets = [], [], [], []
    for i in range(0, len(ksets), batch):
        part = ksets[i:i + batch]
        patches, plc, active, pmask = batcher(torch.from_numpy(np.stack([w for w, _ in part])), model, rng)
        out = model(patches, plc, active, pmask)
        for key in sig:
            sig[key] += out[key].cpu().tolist()
        ks += [k for _, k in part]
        if all(k == 1 for _, k in part) and len(part) > 1:
            pred.append(float(out["pred"]))
            pred_shuf.append(float(model(patches, plc, active, pmask, shuffle_z=True)["pred"]))
            targets.append(out["targets"].reshape(-1, out["targets"].shape[-1])[:512].cpu())
    ks = np.array(ks)
    for key, vals in sig.items():
        v = np.array(vals)
        rec[key] = {int(k): float(v[ks == k].mean()) for k in np.unique(ks)}
        if len(np.unique(ks)) > 1:
            rec[key]["rho"] = float(stats.spearmanr(v, ks).statistic)
    if pred:
        rec["val_pred"] = float(np.mean(pred))
        rec["z_use"] = float(np.mean(pred_shuf) - np.mean(pred))
        rec["target_rank"] = effective_rank(torch.cat(targets))
    model.train()
    return rec


def train(train_roots: list[str], valid_roots: list[str], out: str | Path, cfg: CleaveConfig | None = None,
          steps: int = 50000, batch: int = 4, crop_chunks: int = 64, lr: float = 3e-4, warmup: int = 1000,
          k_schedule: str = "0:1", mono_frac: float = 0.5, beta: float = 1e-3, ema: float = 0.996,
          eval_every: int = 500, eval_k: int = 4, per_k: int = 16, id_chunks: int = 8, min_gap_s: float = 20.0,
          max_val_tracks: int = 200, plateau: int = 8, plateau_delta: float = 0.002, min_z_rank: float = 2.0,
          max_minutes: float = 240.0, workers: int = 4, device: str = "cpu", seed: int = 0,
          limit_frac: float = 1.0, measure: int = 0) -> dict:
    cfg = cfg or CleaveConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_songs = find_songs(train_roots, limit_frac, seed)
    val_songs = find_songs(valid_roots)
    if not train_songs or not val_songs:
        raise SystemExit(f"no songs found: {len(train_songs)} train, {len(val_songs)} valid")
    n_samples = crop_chunks * cfg.chunk_frames * HOP
    model = Cleave(cfg).to(device)
    batcher = Batcher(cfg, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    sampler = KSampler(len(train_songs), parse_schedule(k_schedule), mono_frac, seed)
    loader = torch.utils.data.DataLoader(MixDataset(train_songs, n_samples), batch_size=batch, sampler=sampler,
                                         num_workers=workers, persistent_workers=workers > 0,
                                         pin_memory=device.startswith("cuda"))
    amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {"train_songs": len(train_songs), "valid_songs": len(val_songs), "params": n_params,
            "crop_s": n_samples / SR, "cfg": cfg.to_dict(), "device": device, "amp_bf16": amp}
    _log(json.dumps(meta))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup)))))

    def step_fn(wave: torch.Tensor, step: int) -> dict:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        patches, plc, active, pmask = batcher(wave, model, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o = model(patches, plc, active, pmask, beta=beta)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.ema(1 - (1 - ema) * (math.cos(math.pi * min(1.0, step / steps)) + 1) / 2)
        return o

    if measure:
        return _measure(loader, step_fn, measure, device, n_samples, batch, out)

    crop_id = id_chunks * cfg.chunk_frames * HOP
    pairs = id_pairs(val_songs, crop_id, int(min_gap_s * SR), max_val_tracks, seed)
    ksets = k_sets(val_songs, n_samples, eval_k, per_k, seed)
    _log(json.dumps({"val_tracks": len(pairs), "val_mixes": len(ksets)}))
    hist = (out / "history.jsonl").open("a")

    def record(step: int, extra: dict) -> dict:
        rec = {"step": step, **extra, **evaluate(model, batcher, val_songs, pairs, ksets, crop_id, batch, seed)}
        hist.write(json.dumps(rec) + "\n")
        hist.flush()
        _log(json.dumps(rec))
        return rec

    def save(name: str, step: int) -> None:
        torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step}, out / name)

    rec0 = record(0, {"note": "random init"})
    key = "id_auc_kind" if not math.isnan(rec0.get("id_auc_kind", float("nan"))) else "id_auc"
    best, since, reason = -1.0, 0, f"steps: reached {steps}"
    t0 = time.time()
    run = {"loss": 0.0, "pred": 0.0, "kl": 0.0, "vic_var": 0.0, "n": 0}
    step = 0
    for batch_ in loader:
        step += 1
        sampler.step = step
        o = step_fn(batch_["wave"], step)
        if not torch.isfinite(o["loss"]):
            reason = f"non-finite loss at step {step}"
            break
        for k in ("loss", "pred", "kl", "vic_var"):
            run[k] += float(o[k].detach())
        run["n"] += 1
        if step % eval_every == 0 or step == steps:
            n = max(1, run.pop("n"))
            extra = {k: v / n for k, v in run.items()}
            extra.update({"kmax": sampler.kmax(), "lr": lr_at(step), "minutes": (time.time() - t0) / 60})
            run = {"loss": 0.0, "pred": 0.0, "kl": 0.0, "vic_var": 0.0, "n": 0}
            rec = record(step, extra)
            save("last.pt", step)
            score = rec.get(key, float("nan"))
            if score > best + plateau_delta:
                best, since = score, 0
                save("best.pt", step)
            else:
                since += 1
            if step > warmup and rec.get("z_rank", 99.0) < min_z_rank:
                reason = f"collapse: Z effective rank {rec['z_rank']:.2f} < {min_z_rank} at step {step}"
                break
            if plateau and since >= plateau:
                reason = f"plateau: no {key} gain of {plateau_delta} in {plateau} evaluations (step {step})"
                break
        if step >= steps:
            break
        if (time.time() - t0) / 60 > max_minutes:
            reason = f"max_minutes: {max_minutes} reached at step {step}"
            break
    hist.close()
    summary = {"stop": reason, "step": step, "best_" + key: best, "random_init_" + key: rec0.get(key),
               "minutes": (time.time() - t0) / 60, **meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    _log(json.dumps(summary))
    return summary


def _measure(loader, step_fn, n_steps: int, device: str, n_samples: int, batch: int, out: Path) -> dict:
    """Time ``n_steps`` steps after 3 warm-up ones: data wait vs compute, peak GPU memory, songs per hour."""
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    wait, comp = [], []
    it = iter(loader)
    for step in range(1, n_steps + 4):
        t = time.time()
        b = next(it)
        t1 = time.time()
        step_fn(b["wave"], step)
        if cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        if step > 3:
            wait.append(t1 - t)
            comp.append(t2 - t1)
        _log(json.dumps({"step": step, "wait_s": round(t1 - t, 3), "compute_s": round(t2 - t1, 3)}))
    s_step = float(np.mean(wait) + np.mean(comp))
    rec = {"steps": n_steps, "batch": batch, "crop_s": n_samples / SR, "wait_s": float(np.mean(wait)),
           "compute_s": float(np.mean(comp)), "s_per_step": s_step, "crops_per_hour": 3600 * batch / s_step,
           "audio_hours_per_hour": 3600 * batch * n_samples / SR / 3600 / s_step,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None}
    (out / "measure.json").write_text(json.dumps(rec, indent=1))
    _log(json.dumps(rec))
    return rec
