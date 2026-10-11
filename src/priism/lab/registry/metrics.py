"""How consistent song-wide source ids are, from token labels and the references' energy per token.

A method gives every token (frame x band) a cluster id, the same id meaning the same source over the whole
song. Each reference source owns a share of each token (its power over the total, a soft mask), so a
token-weighted contingency table C[k, r] says how much of reference r cluster k holds. From it:

- **purity**: share of the mass in each cluster's majority reference (low = sources merged);
- **completeness**: share of each reference's mass in its majority cluster (low = a source split, or
  given another id in another part of the song);
- both **global** (whole song) and **local** (within each chunk, then averaged): local completeness only
  asks for clean chunks, global also asks for the same id everywhere. The gap between them is the cost of
  losing track of sources from one chunk to the next, the defect this plan is about;
- **permutation rate**: share of (chunk, reference) mass where the reference's majority cluster in that
  chunk is not its song-wide majority cluster;
- **entries** (online methods): new ids matched to the moments references start playing;
- **V-measure** (Rosenberg & Hirschberg 2007): entropy-based homogeneity and completeness and their
  harmonic mean, global and local. Unlike purity it gives a trivial labelling 0: one cluster for the whole
  song has completeness 1 but homogeneity 0, where purity would still credit it with the majority
  reference's share. Thresholds are tuned on V, never on purity / completeness, which reward merging.

Tokens count equally once audible (above ``floor_db`` under the song's loudest token), whatever their
power: weighting by power would let the kick and the bass decide everything.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def ownership(energy: np.ndarray, groups: np.ndarray | None = None, floor_db: float = 60.0
              ) -> tuple[np.ndarray, np.ndarray]:
    """energy (J, T, Nb) -> (shares (R, T, Nb) summing to 1 where audible, audible (T, Nb) bool).
    ``groups`` (J,) merges sources into references (twins into one instrument)."""
    if groups is not None:
        ids = np.unique(groups)
        energy = np.stack([energy[groups == g].sum(0) for g in ids])
    total = energy.sum(0)
    audible = total > total.max() * 10 ** (-floor_db / 10)
    return energy / np.maximum(total, 1e-20), audible


def contingency(labels: np.ndarray, shares: np.ndarray, audible: np.ndarray) -> np.ndarray:
    """labels (T, Nb) ids >= 0 (-1 = none) -> C (K, R) token mass of reference r under cluster k."""
    keep = audible & (labels >= 0)
    lab = labels[keep]
    K = int(lab.max()) + 1 if lab.size else 0
    C = np.zeros((K, shares.shape[0]))
    for r in range(shares.shape[0]):
        C[:, r] = np.bincount(lab, weights=shares[r][keep], minlength=K)
    return C


def _pur_comp(C: np.ndarray) -> tuple[float, float]:
    tot = C.sum()
    if tot <= 0:
        return np.nan, np.nan
    return float(C.max(1).sum() / tot), float(C.max(0).sum() / tot)


def _vmeasure(C: np.ndarray) -> tuple[float, float, float]:
    """(homogeneity, completeness, V) of a soft contingency table C (K, R)."""
    tot = C.sum()
    if tot <= 0:
        return np.nan, np.nan, np.nan
    P = C / tot
    pk, pr = P.sum(1), P.sum(0)
    nz = P > 0
    h_r = -np.sum(pr[pr > 0] * np.log(pr[pr > 0]))
    h_k = -np.sum(pk[pk > 0] * np.log(pk[pk > 0]))
    h_r_k = -np.sum(P[nz] * np.log(P[nz] / np.broadcast_to(pk[:, None], P.shape)[nz]))
    h_k_r = -np.sum(P[nz] * np.log(P[nz] / np.broadcast_to(pr[None, :], P.shape)[nz]))
    hom = 1.0 if h_r == 0 else 1 - h_r_k / h_r
    com = 1.0 if h_k == 0 else 1 - h_k_r / h_k
    return float(hom), float(com), float(2 * hom * com / (hom + com)) if hom + com > 0 else 0.0


def score(labels: np.ndarray, energy: np.ndarray, chunk: int, groups: np.ndarray | None = None,
          births: list[tuple[int, int]] | None = None, entry_tol: int | None = None, floor_db: float = 60.0
          ) -> dict:
    """All metrics of one song. ``chunk``: frames per chunk; ``births``: (frame, cluster id) of every id the
    method created, for the entry scores; ``entry_tol``: frames of slack for an entry (default one chunk)."""
    shares, audible = ownership(energy, groups, floor_db)
    C = contingency(labels, shares, audible)
    pur, comp = _pur_comp(C)
    hom, vcom, v = _vmeasure(C)
    major = C.argmax(0) if len(C) else np.zeros(shares.shape[0], int)  # song-wide cluster of each reference
    T = labels.shape[0]
    lp, lc, lv, w, perm, perm_w = [], [], [], [], 0.0, 0.0
    for t0 in range(0, T, chunk):
        sl = slice(t0, t0 + chunk)
        Cc = contingency(labels[sl], shares[:, sl], audible[sl])
        if len(Cc) == 0 or Cc.sum() <= 0:
            continue
        p, c = _pur_comp(Cc)
        lp.append(p)
        lc.append(c)
        lv.append(_vmeasure(Cc)[2])
        w.append(Cc.sum())
        mass = Cc.sum(0)
        here = mass > 0.01 * mass.sum()  # references really present in this chunk
        local = Cc.argmax(0)
        perm += float(mass[here & (local != major)].sum())
        perm_w += float(mass[here].sum())
    w = np.asarray(w)
    n_ref = int((C.sum(0) > 0.001 * C.sum()).sum()) if len(C) else 0
    n_clu = int((C.sum(1) > 0.001 * C.sum()).sum()) if len(C) else 0
    out = {"purity": pur, "completeness": comp,
           "local_purity": float(np.average(lp, weights=w)) if len(w) else np.nan,
           "local_completeness": float(np.average(lc, weights=w)) if len(w) else np.nan,
           "permutation_rate": perm / perm_w if perm_w else np.nan,
           "count_error": abs(n_clu - n_ref), "n_clusters": n_clu, "n_refs": n_ref,
           "homogeneity": hom, "v_completeness": vcom, "v_measure": v,
           "local_v_measure": float(np.average(lv, weights=w)) if len(w) else np.nan}
    out["track_loss"] = out["local_completeness"] - out["completeness"]
    if births is not None:
        out.update(entry_scores(births, C, shares, audible, chunk, entry_tol or chunk))
    return out


def entries(shares: np.ndarray, audible: np.ndarray, chunk: int, min_share: float = 0.01) -> np.ndarray:
    """(R,) first frame (chunk start) where each reference owns at least ``min_share`` of the audible
    tokens of a chunk; -1 if never."""
    R, T = shares.shape[0], shares.shape[1]
    first = np.full(R, -1)
    for t0 in range(0, T, chunk):
        a = audible[t0: t0 + chunk]
        if not a.any():
            continue
        frac = (shares[:, t0: t0 + chunk] * a).sum((1, 2)) / a.sum()
        new = (first < 0) & (frac >= min_share)
        first[new] = t0
    return first


def entry_scores(births: list[tuple[int, int]], C: np.ndarray, shares: np.ndarray, audible: np.ndarray,
                 chunk: int, tol: int) -> dict:
    """Precision / recall / F1 of births against reference entries: a birth hits an entry if it is within
    ``tol`` frames and its cluster's majority reference is that reference (each entry hit at most once)."""
    first = entries(shares, audible, chunk)
    owner = C.argmax(1) if len(C) else np.zeros(0, int)
    hit, used = 0, set()
    for f, k in births:
        if k >= len(owner):
            continue
        r = int(owner[k])
        if r not in used and first[r] >= 0 and abs(f - first[r]) <= tol:
            used.add(r)
            hit += 1
    n_true = int((first >= 0).sum())
    p = hit / len(births) if births else np.nan
    r = hit / n_true if n_true else np.nan
    f1 = 2 * p * r / (p + r) if births and n_true and (p + r) > 0 else 0.0
    return {"entry_precision": p, "entry_recall": r, "entry_f1": f1}


def paired_compare(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Method a vs b on the same songs: mean difference with a 95 % bootstrap CI (songs resampled), the
    paired effect size d_z = mean / sd of the differences, and a Wilcoxon signed-rank p-value (no
    normality assumed; ties and zeros handled by scipy's defaults)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else np.nan}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = d.std(ddof=1)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "d_z": float(d.mean() / sd) if sd > 0 else np.nan, "wilcoxon_p": p}
