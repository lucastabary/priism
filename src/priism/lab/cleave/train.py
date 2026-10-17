"""Training loop of the Cleave JEPA, with the lab's guards and the phase 1 measurements built in.

Every evaluation appends one JSON line to ``history.jsonl`` (flushed) and prints it:

- ``id_auc`` / ``id_auc_kind``: Z (posterior mean) of two crops of the same track, ``min_gap_s`` apart, against
  other tracks (all, or of the same kind): the identity criterion C2 of docs/cleave.md. Step 0 is the
  randomly initialised model, the baseline C2 has to beat.
- ``z_rank`` / ``target_rank``: effective ranks of Z and of the target patch tokens (criterion C1); ``z_std``,
  the mean per-dimension std of mu across crops (normalised mu hides a collapse to a constant).
- ``z_use``: validation prediction loss with Z shuffled across songs minus with the true Z (C1: the
  predictor must need Z, else P and L leak the timbre); ``z_use_rel`` = z_use / (loss with shuffled Z), the
  share of the error the true Z removes, comparable across target scales.
- ``pl_r2_z``: out-of-fold ridge R² from Z to the crop's P/L statistics (mean pitch register, mean loudness),
  folds by track: Z should not carry what P and L already give.
- leakage probes on fixed validation views with known random chains (docs/cleave.md 2.9, ``fx_probes``):
  ``fx_r2_z`` (linear Z -> FX parameters, should go to ~0), ``kind_bacc_v`` (V-bar -> instrument kind, should
  stay near ``kind_chance``), with references ``fx_r2_mel`` (the same probe on the plain mean/std log-mel: how
  much effect a linear map can see at all), ``fx_r2_v``, ``kind_bacc_z`` and ``fx_id_auc`` (Z of the same
  track under two chains against other tracks).
- per k in 1..``eval_k``: the three uncertainty signals (posterior sigma, predictor variance, predictor
  error) and their Spearman correlation with k over the validation mixes (docs/cleave.md 2.8).

Guards: non-finite loss, Z collapse (effective rank below ``min_z_rank`` after ``warmup``), a plateau of
``id_auc_kind`` (or ``id_auc`` when kinds are unknown), ``max_minutes``. ``--measure`` only times steps
(data wait vs compute, peak GPU memory) and extrapolates, the step before any long run.

Preemptible: ``last.pt`` holds the whole training state (model with its EMA target, optimizer, step, best
score and plateau counter, RNG, training minutes), written at every evaluation and on SIGTERM (the step in
progress ends, then it saves and stops). With ``resume`` the same command continues from it, so rerunning it
after a stop is idempotent (a finished run stays finished). ``max_minutes`` counts the minutes of all segments.
The data stream of a resumed segment is drawn afresh (sampler seeded by the step): not bit-exact.
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from .data import (FX_COLS, KSampler, MixDataset, PoolBatchSampler, Song, active_frac, collate, find_songs, load_mix,
                   parse_schedule, read_crop, render_fx, sample_fx, song_bytes, song_frames_sr)
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
        self.front = Frontend(cfg.p_mode, cfg.l_mode, cfg.onset_mode).to(device)

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


def fx_set(songs: list[Song], crop: int, max_tracks: int, seed: int = 0, n_chains: int = 4) -> list[tuple]:
    """Fixed probe views: (song, stem, start, kind, [chains]) for playing crops, one crop per track."""
    rng = np.random.default_rng(seed + 1)
    out = []
    for si in rng.permutation(len(songs)):
        song = songs[si]
        total = song_frames_sr(song)
        for st in range(len(song.stems)):
            a = int(rng.integers(0, max(1, total - crop)))
            if active_frac(read_crop(song, st, a, crop)) < 0.25:
                continue
            out.append((int(si), st, a, song.kinds[st], [sample_fx(rng) for _ in range(n_chains)]))
            if len(out) >= max_tracks:
                return out
    return out


def _group_folds(groups: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    """Fold index per row; all rows of a group (the views of a track) share a fold."""
    g = np.unique(groups)
    fold_of = dict(zip(np.random.default_rng(seed).permutation(g), np.arange(len(g)) % n_folds))
    return np.array([fold_of[x] for x in groups])


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float):
    mx, sx, my = X.mean(0), X.std(0) + 1e-8, Y.mean(0)
    Xs = (X - mx) / sx
    W = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ (Y - my))
    return lambda Xn: ((Xn - mx) / sx) @ W + my


def ridge_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_folds: int = 5,
             alphas: tuple = (0.1, 1.0, 10.0, 100.0, 1000.0), seed: int = 0, per_col: bool = False):
    """Out-of-fold R² of a linear (ridge) map X -> Y, mean over the columns of Y that vary (or the R² of
    every column, NaN where it does not vary, with ``per_col``). Folds by group; alpha picked by an inner
    group CV on the training folds only, so the score never sees its test rows."""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    keep = Y.std(0) > 1e-6
    n_cols = Y.shape[1]
    Y = Y[:, keep]
    folds = _group_folds(groups, n_folds, seed)
    pred = np.zeros_like(Y)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if not te.any() or len(np.unique(groups[tr])) < 3:
            return np.full(n_cols, np.nan) if per_col else float("nan")
        inner = _group_folds(groups[tr], 3, seed + 1)
        sse = [sum(((_ridge_fit(X[tr][inner != i], Y[tr][inner != i], a)(X[tr][inner == i]) - Y[tr][inner == i]) ** 2).sum()
                   for i in range(3)) for a in alphas]
        pred[te] = _ridge_fit(X[tr], Y[tr], alphas[int(np.argmin(sse))])(X[te])
    r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    if per_col:
        out = np.full(n_cols, np.nan)
        out[keep] = r2
        return out
    return float(r2.mean())


def _center(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean: what is left varies within a track, i.e. with the chain only."""
    X = X.astype(np.float64).copy()
    for g in np.unique(groups):
        X[groups == g] -= X[groups == g].mean(0)
    return X


def _logreg_fit(X: np.ndarray, y: np.ndarray, n_cls: int, l2: float):
    """Class-balanced multinomial logistic regression (L-BFGS) on standardised X."""
    mx, sx = X.mean(0), X.std(0) + 1e-8
    Xt = torch.from_numpy((X - mx) / sx).float()
    yt = torch.from_numpy(y).long()
    w_cls = torch.from_numpy(len(y) / (n_cls * np.maximum(np.bincount(y, minlength=n_cls), 1))).float()
    lin = torch.nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xt), yt, weight=w_cls) + l2 * (lin.weight ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return lambda Xn: lin(torch.from_numpy((Xn - mx) / sx).float()).argmax(-1).numpy()


def _bacc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def logreg_bacc(X: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_folds: int = 5,
                l2s: tuple = (1e-3, 1e-2, 1e-1), seed: int = 0) -> tuple[float, float]:
    """Out-of-fold balanced accuracy of a linear classifier X -> label, and its chance level (1 / classes:
    the balanced accuracy of always answering the majority class). Classes with fewer than ``n_folds``
    groups are dropped; folds by group, L2 by inner group CV on the training folds."""
    n_groups = {c: len(np.unique(groups[labels == c])) for c in np.unique(labels)}
    ok = np.array([n_groups[c] >= n_folds for c in labels])
    X, labels, groups = X[ok].astype(np.float64), labels[ok], groups[ok]
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan"), float("nan")
    y = np.searchsorted(classes, labels)
    folds = _group_folds(groups, n_folds, seed)
    pred = np.zeros_like(y)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        inner = _group_folds(groups[tr], 3, seed + 1)
        score = [np.mean([_bacc(y[tr][inner == i], _logreg_fit(X[tr][inner != i], y[tr][inner != i], len(classes), l2)(X[tr][inner == i]))
                          for i in range(3) if (inner == i).any()]) for l2 in l2s]
        pred[te] = _logreg_fit(X[tr], y[tr], len(classes), l2s[int(np.argmax(score))])(X[te])
    return _bacc(y, pred), 1.0 / len(classes)


@torch.no_grad()
def fx_probes(model: Cleave, batcher: Batcher, songs: list[Song], fxs: list[tuple], crop: int, batch: int,
              seed: int = 0) -> dict:
    """Leakage probes on the fixed views ``fxs`` (``fx_set``), rendered on the fly (deterministic).

    The FX regressions run on features and chains centred per track: across unseen tracks, identity
    variance would drown the effects (a linear probe cannot tell them apart from a handful of tracks), so
    ``fx_r2_*`` is the share of the within-track chain variance a linear map of the features explains.
    ``fx_r2_z_by`` breaks ``fx_r2_z`` down per effect (mean over its columns)."""
    if len(fxs) < 10:
        return {}
    model.eval()
    zs, vs, mels, fx, kinds, groups = [], [], [], [], [], []
    views = [(i, s, st, a, kind, f) for i, (s, st, a, kind, chains) in enumerate(fxs) for f in chains]
    for j in range(0, len(views), batch):
        part = views[j:j + batch]
        waves = [render_fx(read_crop(songs[s], st, a, crop), f) for _, s, st, a, _, f in part]
        wave = torch.from_numpy(np.stack(waves))
        patches, _, active, _ = batcher(wave, model, None)
        mu, _, _, v = model.encode(patches, active)
        zs.append(mu.float().cpu().numpy())
        if v is not None:
            w = active.float()[..., None]
            vs.append(((v.float() * w).sum(1) / w.sum(1).clamp_min(1)).cpu().numpy())
        logmel, _, _ = batcher.front(wave.to(batcher.device))
        mels.append(torch.cat([logmel.mean(1), logmel.std(1)], -1).cpu().numpy())
        for i, _, _, _, kind, f in part:
            fx.append(f)
            kinds.append(kind)
            groups.append(i)
    Z, F_, M = np.concatenate(zs), np.stack(fx), np.concatenate(mels)
    kinds, groups = np.array(kinds), np.array(groups)
    Fc = _center(F_, groups)
    r2_z = ridge_r2(_center(Z, groups), Fc, groups, seed=seed, per_col=True)
    rec = {"fx_r2_z": float(np.nanmean(r2_z)), "fx_r2_mel": ridge_r2(_center(M, groups), Fc, groups, seed=seed),
           "n_fx_views": len(Z)}
    by: dict = {}
    for c, r in zip(FX_COLS, r2_z):
        by.setdefault(c.split("_")[0], []).append(r)
    rec["fx_r2_z_by"] = {k: float(np.nanmean(v)) for k, v in by.items() if not np.all(np.isnan(v))}
    rec["kind_bacc_z"], rec["kind_chance"] = logreg_bacc(Z, kinds, groups, seed=seed)
    if vs:
        V = np.concatenate(vs)
        rec["fx_r2_v"] = ridge_r2(_center(V, groups), Fc, groups, seed=seed)
        rec["kind_bacc_v"], _ = logreg_bacc(V, kinds, groups, seed=seed)
    zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    n_ch = len(fxs[0][4])
    sim = zn[0::n_ch] @ zn[1::n_ch].T  # chain 0 vs chain 1 of every track
    eye = np.eye(len(sim), dtype=bool)
    rec["fx_id_auc"] = _auc(sim[eye], sim[~eye])
    model.train()
    return rec


@torch.no_grad()
def evaluate(model: Cleave, batcher: Batcher, val_songs: list[Song], pairs: list[tuple], ksets: list, crop: int,
             batch: int, seed: int = 0, fxs: list[tuple] | None = None) -> dict:
    model.eval()
    rec: dict = fx_probes(model, batcher, val_songs, fxs, crop, batch, seed) if fxs else {}
    model.eval()
    # identity: Z of crop A vs crop B of each track
    zs, plstats = [], []
    for i in range(0, len(pairs), batch):
        part = pairs[i:i + batch]
        waves = [read_crop(val_songs[s], st, x, crop) for s, st, a, b, _ in part for x in (a, b)]
        patches, plc, active, _ = batcher(torch.from_numpy(np.stack(waves)), model, None)
        mu, _, _, _ = model.encode(patches, active)
        zs.append(mu.float().cpu())
        frames = plc.reshape(plc.shape[0], -1, model.cfg.pl_dim).float().cpu()
        sal = frames[..., : model.cfg.pl_dim - 2]
        register = (sal * torch.arange(sal.shape[-1])).sum(-1) / sal.sum(-1).clamp_min(1e-6)
        played = sal.sum(-1) > 0
        plstats.append(torch.stack([(register * played).sum(1) / played.sum(1).clamp_min(1),
                                    frames[..., -2].mean(1)], -1))
    if zs:
        raw = torch.cat(zs)
        rec["z_std"] = float(raw.std(0).mean())  # a constant mu has a high rank once normalised, not a std
        z = torch.nn.functional.normalize(raw, dim=-1).view(-1, 2, raw.shape[-1])
        za, zb = z[:, 0], z[:, 1]
        sim = (za @ zb.T).numpy()
        kinds = np.array([p[4] for p in pairs])
        eye = np.eye(len(pairs), dtype=bool)
        same_kind = kinds[:, None] == kinds[None, :]
        rec["id_auc"] = _auc(sim[eye], sim[~eye])
        rec["id_auc_kind"] = _auc(sim[eye], sim[~eye & same_kind])
        rec["z_rank"] = effective_rank(torch.cat([za, zb]))
        rec["n_tracks"] = len(pairs)
        groups = np.repeat(np.arange(len(pairs)), 2)
        rec["pl_r2_z"] = ridge_r2(raw.numpy(), torch.cat(plstats).numpy(), groups, seed=seed)
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
        rec["z_use_rel"] = rec["z_use"] / max(float(np.mean(pred_shuf)), 1e-12)
        rec["target_rank"] = effective_rank(torch.cat(targets))
    model.train()
    return rec


def train(train_roots: list[str], valid_roots: list[str], out: str | Path, cfg: CleaveConfig | None = None,
          steps: int = 50000, batch: int = 4, crop_chunks: int = 64, lr: float = 3e-4, warmup: int = 1000,
          k_schedule: str = "0:1", mono_frac: float = 0.5, beta: float = 1e-3, ema: float = 0.996,
          eval_every: int = 500, eval_k: int = 4, per_k: int = 16, id_chunks: int = 8, min_gap_s: float = 20.0,
          max_val_tracks: int = 200, plateau: int = 8, plateau_delta: float = 0.002, min_z_rank: float = 2.0,
          max_minutes: float = 240.0, workers: int = 4, device: str = "cpu", seed: int = 0,
          limit_frac: float = 1.0, measure: int = 0, pair_frac: float = 0.0, w_inv: float = 1.0,
          w_swap: float = 1.0, fx_tracks: int | None = None, fx_chains: int = 4, other_view: bool = False,
          v_drop: float = 0.0, w_vinv: float = 0.0, inv_mode: str = "nce", inv_tau: float = 0.1,
          pool_gb: float = 0.0, pool_reuse: int = 32, w_var: float = 1.0, w_cov: float = 0.04,
          w_ctx: float = 1.0, resume: bool = False, w_zuse: float = 0.0, zuse_margin: float = 0.1) -> dict:
    cfg = cfg or CleaveConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    state = _resume_state(out, cfg, device) if resume and not measure else None
    if state and state.get("done"):
        _log(json.dumps({"resume": f"already finished ({state['done']}), nothing to do"}))
        return json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    train_songs = find_songs(train_roots, limit_frac, seed)
    val_songs = find_songs(valid_roots)
    if not train_songs or not val_songs:
        raise SystemExit(f"no songs found: {len(train_songs)} train, {len(val_songs)} valid")
    n_samples = crop_chunks * cfg.chunk_frames * HOP
    model = Cleave(cfg).to(device)
    batcher = Batcher(cfg, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    start = 0
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        rng.bit_generator.state = state["np_rng"]
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and device.startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng"], torch.device(device))
        start = state["step"]
        seed_data = int(np.random.SeedSequence([seed, start]).generate_state(1)[0])
        _log(json.dumps({"resume": f"from {out / 'last.pt'} at step {start}"}))
    else:
        seed_data = seed
    pool = {}
    if pool_gb > 0:  # RAM song pools, one per worker (data.py): the budget sets the songs per pool
        pools = max(1, workers)
        mean_bytes = float(np.mean([song_bytes(s) for s in train_songs]))
        pool_songs = max(1, int(pool_gb * 1e9 / (pools * mean_bytes)))
        sampler = PoolBatchSampler(len(train_songs), batch, parse_schedule(k_schedule), pools, pool_songs,
                                   pool_reuse, mono_frac, seed_data)
        dataset = MixDataset(train_songs, n_samples, pair_frac, other_view, cache_songs=pool_songs + 1)
        loader_args = {"batch_sampler": sampler}
        pool = {"pool_songs": pool_songs, "pool_reuse": pool_reuse, "pool_gb_est": pools * (pool_songs + 1) * mean_bytes / 1e9}
    else:
        sampler = KSampler(len(train_songs), parse_schedule(k_schedule), mono_frac, seed_data)
        dataset = MixDataset(train_songs, n_samples, pair_frac, other_view)
        loader_args = {"batch_size": batch, "sampler": sampler}
    loader = torch.utils.data.DataLoader(dataset, collate_fn=collate, num_workers=workers, persistent_workers=workers > 0,
                                         pin_memory=device.startswith("cuda"), **loader_args)
    amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {"train_songs": len(train_songs), "valid_songs": len(val_songs), "params": n_params,
            "crop_s": n_samples / SR, "cfg": cfg.to_dict(), "device": device, "amp_bf16": amp,
            "pair_frac": pair_frac, "w_inv": w_inv, "w_swap": w_swap, "other_view": other_view, "v_drop": v_drop,
            "w_vinv": w_vinv, "inv_mode": inv_mode, "inv_tau": inv_tau, "ema": ema, "w_var": w_var, "w_cov": w_cov,
            "w_ctx": w_ctx, "w_zuse": w_zuse, "zuse_margin": zuse_margin,
            **pool}
    _log(json.dumps(meta))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup)))))

    def step_fn(b: dict, step: int) -> dict:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        patches, plc, active, pmask = batcher(b["wave"], model, rng)
        pair = None
        if len(b["pair_idx"]):
            patches_b, _, active_b, _ = batcher(b["wave_b"], model, None)
            pair = (b["pair_idx"].to(device), patches_b, active_b)
            if "wave_o" in b:
                patches_o, _, active_o, _ = batcher(b["wave_o"], model, None)
                pair = pair + (patches_o, active_o)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o = model(patches, plc, active, pmask, beta=beta, w_var=w_var, w_cov=w_cov, pair=pair, w_inv=w_inv,
                      w_swap=w_swap, v_drop=v_drop, w_vinv=w_vinv, inv_mode=inv_mode, inv_tau=inv_tau,
                      w_ctx=w_ctx, w_zuse=w_zuse, zuse_margin=zuse_margin)
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
    if fx_tracks is None:  # auto: probes for V-bar or paired runs; phase 1a runs keep their eval cost
        fx_tracks = 128 if cfg.v_dim or pair_frac > 0 else 0
    fxs = fx_set(val_songs, crop_id, fx_tracks, seed, fx_chains) if fx_tracks else []
    _log(json.dumps({"val_tracks": len(pairs), "val_mixes": len(ksets), "fx_tracks": len(fxs)}))
    hist = (out / "history.jsonl").open("a")

    def record(step: int, extra: dict) -> dict:
        rec = {"step": step, **extra, **evaluate(model, batcher, val_songs, pairs, ksets, crop_id, batch, seed, fxs)}
        hist.write(json.dumps(rec) + "\n")
        hist.flush()
        _log(json.dumps(rec))
        return rec

    def save(name: str, step: int, full: bool = False, done: str | None = None) -> None:
        ckpt = {"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step}
        if full:  # everything --resume needs (see the module docstring)
            ckpt.update(opt=opt.state_dict(), best=best, since=since, key=key, rec0=rec0, run=run,
                        minutes=(time.time() - t0) / 60, np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(torch.device(device)) if device.startswith("cuda") else None,
                        done=done)
        tmp = out / f"{name}.tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, out / name)  # a kill while saving leaves the previous checkpoint whole

    keys = ("loss", "pred", "kl", "vic_var", "ctx", "zgap", "base", "inv", "swap", "vinv")
    if state:
        key, rec0, best, since = state["key"], state["rec0"], state["best"], state["since"]
        run = {**dict.fromkeys(keys, 0.0), **state["run"]}
        t0 = time.time() - state["minutes"] * 60
    else:
        rec0 = record(0, {"note": "random init"})
        key = "id_auc_kind" if not math.isnan(rec0.get("id_auc_kind", float("nan"))) else "id_auc"
        rec0 = {key: rec0.get(key)}
        best, since = -1.0, 0
        t0 = time.time()
        run = {**dict.fromkeys(keys, 0.0), "n": 0}
    reason = f"steps: reached {steps}"
    terminated = []
    try:  # SIGTERM (the agent preempting or stopping the job): end the step, save last.pt, stop
        old_term = signal.signal(signal.SIGTERM, lambda *_: terminated.append(1))
    except ValueError:  # not the main thread
        old_term = None
    step = sampler.step = start
    for batch_ in loader if start < steps else []:
        step += 1
        sampler.step = step
        o = step_fn(batch_, step)
        if not torch.isfinite(o["loss"]):
            reason = f"non-finite loss at step {step}"
            break
        for k in keys:
            run[k] += float(o[k].detach())
        run["n"] += 1
        if step % eval_every == 0 or step == steps:
            n = max(1, run.pop("n"))
            extra = {k: v / n for k, v in run.items()}
            extra.update({"kmax": sampler.kmax(), "lr": lr_at(step), "minutes": (time.time() - t0) / 60})
            run = {**dict.fromkeys(keys, 0.0), "n": 0}
            rec = record(step, extra)
            score = rec.get(key, float("nan"))
            if score > best + plateau_delta:
                best, since = score, 0
                save("best.pt", step)
            else:
                since += 1
            save("last.pt", step, full=True)
            if step > warmup and rec.get("z_rank", 99.0) < min_z_rank:
                reason = f"collapse: Z effective rank {rec['z_rank']:.2f} < {min_z_rank} at step {step}"
                break
            if plateau and since >= plateau:
                reason = f"plateau: no {key} gain of {plateau_delta} in {plateau} evaluations (step {step})"
                break
        if terminated:
            reason = f"terminated (SIGTERM) at step {step}, resumable"
            break
        if step >= steps:
            break
        if (time.time() - t0) / 60 > max_minutes:
            reason = f"max_minutes: {max_minutes} reached at step {step}"
            break
    if old_term is not None:
        signal.signal(signal.SIGTERM, old_term)
    if not reason.startswith("non-finite"):  # NaN weights never replace the last good checkpoint
        save("last.pt", step, full=True, done=None if terminated else reason)
    hist.close()
    summary = {"stop": reason, "step": step, "best_" + key: best, "random_init_" + key: rec0.get(key),
               "minutes": (time.time() - t0) / 60, **meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    _log(json.dumps(summary))
    return summary


def _resume_state(out: Path, cfg: CleaveConfig, device: str) -> dict | None:
    """The full state in ``out/last.pt`` for ``--resume``, or None to start fresh (no checkpoint yet)."""
    path = out / "last.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location=device, weights_only=False)
    if "opt" not in state:
        raise SystemExit(f"{path} was saved before --resume existed (no optimizer state): use a new --out")
    if CleaveConfig(**state["cfg"]) != cfg:
        raise SystemExit(f"{path} has another model config than this command: use a new --out")
    return state


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
        step_fn(b, step)
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


def probe_checkpoint(ckpt: str | Path, valid_roots: list[str], id_chunks: int = 8, fx_tracks: int = 128,
                     batch: int = 4, device: str = "cpu", seed: int = 0, fx_chains: int = 4) -> dict:
    """The leakage probes alone on a saved checkpoint (e.g. a phase 1a no-V run, the baseline of 2.9). Same
    views as in training for the same ``valid_roots``, ``id_chunks``, ``fx_tracks`` and ``seed``."""
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = CleaveConfig(**state["cfg"])
    model = Cleave(cfg).to(device)
    model.load_state_dict(state["model"])
    songs = find_songs(valid_roots)
    crop = id_chunks * cfg.chunk_frames * HOP
    rec = {"ckpt": str(ckpt), "step": state.get("step"), "v_dim": cfg.v_dim,
           **fx_probes(model, Batcher(cfg, device), songs, fx_set(songs, crop, fx_tracks, seed, fx_chains), crop, batch,
                       seed)}
    _log(json.dumps(rec))
    return rec
