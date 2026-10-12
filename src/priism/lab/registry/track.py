"""Ways to give every token of a song a song-wide source id, from token embeddings alone.

Embeddings are (T, Nb, k): the cached trunk tokens (phase 1) or the identity head's output (phase 2).
``active`` (T, Nb) marks the audible tokens (from the mix's power, known at inference); the others get -1.

- ``per_chunk``: cluster each chunk on its own, new ids every chunk. What a 4 s separator does: clean
  chunks, no link between them. Lower bound of song-wide consistency.
- ``per_chunk_linked``: same, then each chunk's clusters are matched to the previous chunk's (Hungarian
  on centroid cosine), as ``split.py`` links windows. Errors accumulate along the chain.
- ``whole_song``: cluster all tokens of the song at once (micro k-means, then average-linkage on the
  micro centroids), plan-v2's discovery. Sees everything, ignores order.
- ``online_registry``: read the song in order. Tokens close to a known source join it (its centroid
  follows them slowly); what nothing in the registry explains is clustered, and each cluster big enough is
  born as a new source (a birth, scored against the references' entries). With ``two_pass``, every token
  is then labelled again against the final registry: a source that entered at minute 3 is also found at
  minute 1, where it may have been quiet.

All clustering is cosine (embeddings are L2-normalised first) and is cut by a distance threshold, so the
number of sources is never given; ``n`` variants (oracle counts) exist only as upper bounds.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment


def normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def spherical_kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """x (N, d) unit rows -> (centroids (k', d), assignment (N,)); empty clusters dropped."""
    k = min(k, len(x))
    # k-means++ seeding on cosine distance: spread-out starts, fewer bad local minima than random rows.
    c = [x[rng.integers(len(x))]]
    d = 1 - x @ c[0]
    for _ in range(k - 1):
        p = np.maximum(d, 0) ** 2
        if p.sum() <= 0:
            break
        c.append(x[rng.choice(len(x), p=p / p.sum())])
        d = np.minimum(d, 1 - x @ c[-1])
    c = np.stack(c)
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        new = np.zeros_like(c)
        np.add.at(new, a, x)
        keep = np.bincount(a, minlength=len(c)) > 0
        c = normalise(new[keep])
    a = (x @ c.T).argmax(1)
    return c, a


def cluster(x: np.ndarray, rng: np.random.Generator, threshold: float | None = None, n: int | None = None,
            micro: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Unit rows (N, d) -> (centroids (K, d), assignment (N,)): micro k-means, then average linkage on the
    micro centroids cut at cosine distance ``threshold`` (or into ``n`` clusters)."""
    if len(x) == 0:
        return np.zeros((0, x.shape[1])), np.zeros(0, int)
    mc, ma = spherical_kmeans(x, min(micro, max(1, len(x) // 8)), rng)
    if len(mc) == 1:
        return mc, np.zeros(len(x), int)
    z = linkage(mc, method="average", metric="cosine")
    g = fcluster(z, n, criterion="maxclust") - 1 if n else fcluster(z, threshold, criterion="distance") - 1
    a = g[ma]
    cent = normalise(np.stack([x[a == j].sum(0) for j in range(g.max() + 1)]))
    return cent, a


def _labels_like(active: np.ndarray) -> np.ndarray:
    return np.full(active.shape, -1, int)


def per_chunk(emb: np.ndarray, active: np.ndarray, chunk: int, threshold: float, seed: int = 0) -> np.ndarray:
    x, rng = normalise(emb), np.random.default_rng(seed)
    out, nxt = _labels_like(active), 0
    for t0 in range(0, len(x), chunk):
        sl = slice(t0, t0 + chunk)
        a = active[sl]
        if not a.any():
            continue
        _, lab = cluster(x[sl][a], rng, threshold)
        block = out[sl]
        block[a] = lab + nxt
        nxt += lab.max() + 1
    return out


def per_chunk_linked(emb: np.ndarray, active: np.ndarray, chunk: int, threshold: float, seed: int = 0,
                     link_min: float = 0.5) -> np.ndarray:
    """Chunk clusters matched to the previous chunk's; a match needs cosine >= ``link_min``."""
    x, rng = normalise(emb), np.random.default_rng(seed)
    out, nxt, prev = _labels_like(active), 0, None  # prev: (centroids, ids)
    for t0 in range(0, len(x), chunk):
        sl = slice(t0, t0 + chunk)
        a = active[sl]
        if not a.any():
            continue
        cent, lab = cluster(x[sl][a], rng, threshold)
        ids = np.full(len(cent), -1)
        if prev is not None:
            sim = cent @ prev[0].T
            rows, cols = linear_sum_assignment(-sim)
            for i, j in zip(rows, cols):
                if sim[i, j] >= link_min:
                    ids[i] = prev[1][j]
        for i in np.where(ids < 0)[0]:
            ids[i], nxt = nxt, nxt + 1
        block = out[sl]
        block[a] = ids[lab]
        prev = (cent, ids)
    return out


def whole_song(emb: np.ndarray, active: np.ndarray, threshold: float | None = None, n: int | None = None,
               seed: int = 0, max_tokens: int = 50000, micro: int = 256) -> np.ndarray:
    x, rng = normalise(emb), np.random.default_rng(seed)
    out = _labels_like(active)
    xa = x[active]
    if len(xa) == 0:
        return out
    sub = xa[rng.choice(len(xa), min(len(xa), max_tokens), replace=False)]
    cent, _ = cluster(sub, rng, threshold, n, micro=micro)
    out[active] = (xa @ cent.T).argmax(1)
    return out


def online_registry(emb: np.ndarray, active: np.ndarray, chunk: int, tau: float, min_mass: float = 0.03,
                    ema: float = 0.1, two_pass: bool = True, seed: int = 0
                    ) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """-> (labels (T, Nb), births [(frame, id)], registry centroids (K, d)).

    ``tau``: cosine a token needs to join a known source; ``min_mass``: share of a chunk's audible tokens a
    new cluster needs to be born (smaller unexplained bits join their nearest source); ``ema``: how fast a
    centroid follows its tokens (sounds drift: filters open, effects change)."""
    x, rng = normalise(emb), np.random.default_rng(seed)
    out = _labels_like(active)
    reg = np.zeros((0, x.shape[-1]))
    births: list[tuple[int, int]] = []
    for t0 in range(0, len(x), chunk):
        sl = slice(t0, t0 + chunk)
        a = active[sl]
        if not a.any():
            continue
        xc = x[sl][a]
        lab = np.full(len(xc), -1)
        if len(reg):
            sim = xc @ reg.T
            best = sim.argmax(1)
            known = sim[np.arange(len(xc)), best] >= tau
            lab[known] = best[known]
        rest = np.where(lab < 0)[0]
        if len(rest) >= min_mass * len(xc):
            cent, sub = cluster(xc[rest], rng, 1 - tau)
            for j in range(len(cent)):
                members = rest[sub == j]
                if len(members) >= min_mass * len(xc):
                    births.append((t0, len(reg)))
                    lab[members] = len(reg)
                    reg = np.vstack([reg, cent[j]])
        left = lab < 0
        if left.any() and len(reg):
            lab[left] = (xc[left] @ reg.T).argmax(1)
        for k in np.unique(lab[lab >= 0]):
            reg[k] = normalise((1 - ema) * reg[k] + ema * normalise(xc[lab == k].mean(0)))
        block = out[sl]
        block[a] = lab
    if two_pass and len(reg):
        out[active] = (x[active] @ reg.T).argmax(1)
    return out, births, reg


def merge_coactive(labels: np.ndarray, bin_frames: int = 21, threshold: float | None = None, n: int | None = None
                   ) -> np.ndarray:
    """Merge clusters that play together over the song (temporal co-activation), for scoring against grouped
    references: a head trained on fine synthetic identity splits a "Drums" stem into kick / hats / snare,
    which are always active together. Each cluster's activity is its share of tokens per bin of
    ``bin_frames`` frames (~1 s); clusters are merged by average linkage on 1 - correlation of those curves,
    cut at ``threshold`` or into ``n`` groups (n = the number of references is an oracle: an upper bound)."""
    K = int(labels.max()) + 1 if (labels >= 0).any() else 0
    if K < 2:
        return labels.copy()
    T = labels.shape[0]
    n_bins = -(-T // bin_frames)
    act = np.zeros((K, n_bins))
    for b in range(n_bins):
        lab = labels[b * bin_frames:(b + 1) * bin_frames]
        lab = lab[lab >= 0]
        if lab.size:
            act[:, b] = np.bincount(lab, minlength=K) / lab.size
    live = act.std(1) > 0
    groups = np.arange(K)
    if live.sum() >= 2:
        z = linkage(act[live], method="average", metric="correlation")
        g = fcluster(z, n, criterion="maxclust") if n else fcluster(z, threshold, criterion="distance")
        groups[live] = K + g  # merged ids; constant (always / never active) clusters keep their own
    _, groups = np.unique(groups, return_inverse=True)
    out = labels.copy()
    out[labels >= 0] = groups[labels[labels >= 0]]
    return out
