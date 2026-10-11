"""Identity head (phase 2): token embeddings in which a source keeps one place over the whole song.

Trained on the feature cache only (no trunk, no audio), so a step costs almost nothing and several seeds
fit in the budget; it even runs on a CPU. Each step takes, from each of a few songs, two chunks far apart
in the song and samples tokens that one reference clearly owns (share >= ``own``), the same number per
reference so loud sources do not dominate. A supervised contrastive loss (Khosla et al. 2020) pulls
together tokens of the same source, in the same chunk or a minute apart (other effects, other notes), and
pushes apart the other sources of the song and of the other songs. ``level="instrument"`` counts twins as
one identity (the timbre, Z), ``"track"`` as two (needs what they play, which tokens alone may not hold).

Guards (lab rule: never leave a long run unwatched): the run stops on a non-finite loss, when the
validation score has not improved by ``plateau_delta`` for ``plateau`` evaluations, or at ``max_minutes``.
Every evaluation is appended to ``history.jsonl``; ``best.pt`` keeps the best validation score.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import Cached
from .metrics import ownership, score
from .track import whole_song


class IdentityHead(nn.Module):
    """(T, Nb, k) cached tokens -> (T, Nb, dim) unit embeddings. A band embedding says where a token sits in
    frequency; ``context`` > 0 adds a depthwise convolution over that many frames on each side, so a
    token sees the notes around it."""

    def __init__(self, k: int, n_bands: int, dim: int = 64, hidden: int = 256, context: int = 2):
        super().__init__()
        self.cfg = {"k": k, "n_bands": n_bands, "dim": dim, "hidden": hidden, "context": context}
        self.band = nn.Parameter(torch.zeros(n_bands, k))
        self.conv = nn.Conv1d(k * n_bands, k * n_bands, 2 * context + 1, padding=context,
                              groups=k * n_bands) if context else None
        self.mlp = nn.Sequential(nn.LayerNorm(k), nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:  # (T, Nb, k) -> conv along T, one filter per (band, feature)
            T, Nb, k = x.shape
            x = x + self.conv(x.reshape(T, Nb * k).T[None])[0].T.reshape(T, Nb, k)
        return F.normalize(self.mlp(x + self.band), dim=-1)

    @torch.no_grad()
    def embed(self, tokens: np.ndarray) -> np.ndarray:
        return self(torch.from_numpy(tokens)).numpy()


def _refs(song: Cached, level: str) -> tuple[np.ndarray, np.ndarray]:
    groups = song.instruments if level == "instrument" else None
    return ownership(song.energy, groups)


def sample_tokens(song: Cached, shares: np.ndarray, audible: np.ndarray, chunk: int, min_gap: int,
                  per_ref: int, own: float, rng: np.random.Generator) -> list[tuple[int, int, np.ndarray]]:
    """Two chunks at least ``min_gap`` frames apart (or as far as the song allows) -> for each, the chunk
    start and, per clearly owned token, (reference, frame, band) rows: [(t0, ref, rows (n, 2))]."""
    T = song.tokens.shape[0]
    hi = max(1, T - chunk)
    a = int(rng.integers(hi))
    far = [t for t in range(0, hi, max(1, chunk // 4)) if abs(t - a) >= min(min_gap, hi // 2)]
    b = int(rng.choice(far)) if far else int(rng.integers(hi))
    out = []
    for t0 in (a, b):
        sh = shares[:, t0: t0 + chunk]
        top, arg = sh.max(0), sh.argmax(0)
        ok = audible[t0: t0 + chunk] & (top >= own)
        for r in np.unique(arg[ok]):
            rows = np.argwhere(ok & (arg == r))
            if len(rows) >= 4:
                out.append((t0, int(r), rows[rng.choice(len(rows), min(per_ref, len(rows)), replace=False)]))
    return out


def supcon(z: torch.Tensor, labels: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss over unit rows z (N, d) with integer labels (N,)."""
    sim = z @ z.T / temp
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = pos.any(1)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def validate(head: IdentityHead, songs: list[Cached], level: str, threshold: float, chunk: int) -> dict:
    """Mean song-wide purity / completeness of whole-song clustering on the head's embeddings, and their
    harmonic mean ``f`` (the early-stopping score)."""
    rows = []
    for s in songs:
        groups = s.instruments if level == "instrument" else None
        shares, audible = ownership(s.energy, groups)
        lab = whole_song(head.embed(s.tokens), audible, threshold)
        rows.append(score(lab, s.energy, chunk, groups))
    p = float(np.nanmean([r["purity"] for r in rows]))
    c = float(np.nanmean([r["completeness"] for r in rows]))
    return {"purity": p, "completeness": c, "f": 2 * p * c / (p + c) if p + c > 0 else 0.0,
            "permutation_rate": float(np.nanmean([r["permutation_rate"] for r in rows]))}


def train_head(train: list[Cached], valid: list[Cached], out: str | Path, level: str = "instrument",
               steps: int = 5000, songs_per_step: int = 8, per_ref: int = 32, own: float = 0.6,
               chunk_s: float = 4.0, min_gap_s: float = 20.0, lr: float = 1e-3, temp: float = 0.1,
               dim: int = 64, context: int = 2, eval_every: int = 250, threshold: float = 0.5,
               plateau: int = 6, plateau_delta: float = 0.002, max_minutes: float = 120.0, seed: int = 0,
               log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T0, Nb, k = train[0].tokens.shape
    fs = train[0].frame_s
    chunk, min_gap = max(1, round(chunk_s / fs)), round(min_gap_s / fs)
    head = IdentityHead(k, Nb, dim=dim, context=context)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    refs = [_refs(s, level) for s in train]
    best, since, start, hist = -1.0, 0, time.time(), out / "history.jsonl"
    hist.unlink(missing_ok=True)
    losses: list[float] = []
    reason = "steps"
    for step in range(1, steps + 1):
        zs, labels = [], []
        for si in rng.choice(len(train), min(songs_per_step, len(train)), replace=False):
            song = train[si]
            for t0, r, rows in sample_tokens(song, *refs[si], chunk, min_gap, per_ref, own, rng):
                x = torch.from_numpy(song.tokens[t0: t0 + chunk])
                z = head(x)
                zs.append(z[rows[:, 0], rows[:, 1]])
                labels.append(torch.full((len(rows),), int(si) * 1000 + r))
        if len(zs) < 2:
            continue
        loss = supcon(torch.cat(zs), torch.cat(labels), temp)
        if not torch.isfinite(loss):
            reason = f"non-finite loss at step {step}"
            log(reason)
            break
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if step % eval_every == 0 or step == steps:
            head.eval()
            v = validate(head, valid, level, threshold, chunk)
            head.train()
            rec = {"step": step, "loss": float(np.mean(losses)), "minutes": (time.time() - start) / 60, **v}
            losses = []
            with hist.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            log(json.dumps(rec))
            if v["f"] > best + plateau_delta:
                best, since = v["f"], 0
                torch.save({"cfg": head.cfg, "state": head.state_dict(), "step": step, "valid": v}, out / "best.pt")
            else:
                since += 1
            if plateau and since >= plateau:
                reason = f"plateau: no gain of {plateau_delta} in {plateau} evaluations (step {step})"
                log(reason)
                break
            if rec["minutes"] > max_minutes:
                reason = f"time cap {max_minutes} min (step {step})"
                log(reason)
                break
    torch.save({"cfg": head.cfg, "state": head.state_dict(), "step": step}, out / "last.pt")
    summary = {"stopped": reason, "best_f": best, "level": level, "seed": seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_head(path: str | Path) -> IdentityHead:
    d = torch.load(path, map_location="cpu", weights_only=False)
    head = IdentityHead(**d["cfg"])
    head.load_state_dict(d["state"])
    return head.eval()
