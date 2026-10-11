"""Domain probe: how easily a linear classifier tells two sets of cached songs apart (e.g. synthetic vs NI).

A tracking metric for the generator: the more its songs look like real music to the frozen trunk, the
closer to 0.5 the AUC. Tokens: up to ``n`` audible tokens per song (optionally with a one-hot band);
logistic regression (full-batch LBFGS, L2, classes balanced by weights); 5 folds split by song, so no
song is in both training and test. Reported: held-out token AUC (mean ± sd over folds) and song AUC (mean
probability per held-out song). Part of a gap can be content (genres a set never has), not only the
generator's signature.
"""

from __future__ import annotations

import numpy as np
import torch

from .features import Cached
from .metrics import ownership


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def _song_tokens(s: Cached, n: int, band: bool, rng: np.random.Generator) -> np.ndarray:
    _, audible = ownership(s.energy)
    idx = np.argwhere(audible)
    idx = idx[rng.choice(len(idx), min(n, len(idx)), replace=False)]
    x = s.tokens[idx[:, 0], idx[:, 1]]
    if band:
        x = np.concatenate([x, np.eye(s.tokens.shape[1], dtype=np.float32)[idx[:, 1]]], 1)
    return x


def _fit(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> torch.nn.Linear:
    X, y, w = (torch.from_numpy(np.ascontiguousarray(a, np.float32)) for a in (X, y, w))
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(lin(X)[:, 0], y, weight=w)
                + 1e-3 * lin.weight.pow(2).sum())
        loss.backward()
        return loss.detach()

    opt.step(closure)
    return lin


def probe(a: list[Cached], b: list[Cached], n: int = 1500, band: bool = False, folds: int = 5,
          seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    songs = [(_song_tokens(s, n, band, rng), 0) for s in a] + [(_song_tokens(s, n, band, rng), 1) for s in b]
    tok, song_y, song_p = [], [], []
    # Folds stratified by set: each holds songs of both sets, so every fold has a defined AUC.
    split = [[] for _ in range(folds)]
    for label in (0, 1):
        idx = rng.permutation([i for i, s in enumerate(songs) if s[1] == label])
        for j, i in enumerate(idx):
            split[j % folds].append(int(i))
    for f in split:
        held = set(f)
        X = np.concatenate([songs[i][0] for i in range(len(songs)) if i not in held])
        y = np.concatenate([np.full(len(songs[i][0]), songs[i][1]) for i in range(len(songs)) if i not in held])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        w = np.where(y == 1, 0.5 / (y == 1).mean(), 0.5 / (y == 0).mean())
        lin = _fit((X - mu) / sd, y, w)
        ys, ps = [], []
        for i in f:
            x, label = songs[i]
            with torch.no_grad():
                p = torch.sigmoid(lin(torch.from_numpy(((x - mu) / sd).astype(np.float32)))[:, 0]).numpy()
            ys.append(np.full(len(p), label))
            ps.append(p)
            song_y.append(label)
            song_p.append(float(p.mean()))
        tok.append(_auc(np.concatenate(ys), np.concatenate(ps)))
    return {"token_auc": float(np.mean(tok)), "token_auc_sd": float(np.std(tok, ddof=1)),
            "song_auc": _auc(np.array(song_y), np.array(song_p)), "songs_a": len(a), "songs_b": len(b),
            "band": band}
