"""Phase 1 / 2 comparison: every tracking method on the same songs, thresholds tuned on a dev set only.

Protocol: each method's threshold is chosen on the ``tune`` songs (best mean ``f``, the harmonic mean of
song-wide purity and completeness), then frozen and scored on the ``test`` songs, which never influence
any choice. Each test song gives one paired observation per method; methods are compared to a reference
method with a bootstrap CI over songs, the paired effect size d_z and a Wilcoxon test (Holm-corrected
over the comparisons). Oracle-count variants are reported as upper bounds, never tuned.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .features import Cached
from .metrics import ownership, paired_compare, score
from .track import online_registry, per_chunk, per_chunk_linked, whole_song

METHODS = ("per_chunk", "per_chunk_linked", "whole_song", "online", "online_two_pass")
GRID = {"per_chunk": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "per_chunk_linked": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "whole_song": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        # online: tau, the cosine to join a known source (its new clusters are cut at 1 - tau)
        "online": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "online_two_pass": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}


def run_method(name: str, emb: np.ndarray, audible: np.ndarray, chunk: int, param: float, seed: int = 0
               ) -> tuple[np.ndarray, list | None]:
    if name == "per_chunk":
        return per_chunk(emb, audible, chunk, param, seed), None
    if name == "per_chunk_linked":
        return per_chunk_linked(emb, audible, chunk, param, seed), None
    if name == "whole_song":
        return whole_song(emb, audible, param, seed=seed), None
    if name in ("online", "online_two_pass"):
        lab, births, _ = online_registry(emb, audible, chunk, param, two_pass=name == "online_two_pass", seed=seed)
        return lab, births
    raise ValueError(name)


def song_scores(song: Cached, embed: Callable[[np.ndarray], np.ndarray], method: str, param: float, chunk: int,
                level: str, seed: int = 0) -> dict:
    groups = song.instruments if level == "instrument" else None
    _, audible = ownership(song.energy, groups)
    lab, births = run_method(method, embed(song.tokens), audible, chunk, param, seed)
    return score(lab, song.energy, chunk, groups, births=births)


def _f(rows: list[dict]) -> float:
    p, c = np.nanmean([r["purity"] for r in rows]), np.nanmean([r["completeness"] for r in rows])
    return float(2 * p * c / (p + c)) if p + c > 0 else 0.0


def tune(songs: list[Cached], embed, method: str, chunk: int, level: str, grid: list[float] | None = None,
         log=print) -> tuple[float, dict]:
    grid = grid or GRID[method]
    res = {g: _f([song_scores(s, embed, method, g, chunk, level) for s in songs]) for g in grid}
    best = max(res, key=res.get)
    log(f"tune {method} ({level}): " + ", ".join(f"{g}: {v:.3f}" for g, v in res.items()) + f" -> {best}")
    return best, res


def compare(tune_songs: list[Cached], test_songs: list[Cached], embed, chunk: int, level: str = "track",
            methods=METHODS, reference: str = "online_two_pass", log=print) -> dict:
    """Tune each method on ``tune_songs``, score on ``test_songs`` -> per-song rows, means and paired tests."""
    params = {m: tune(tune_songs, embed, m, chunk, level, log=log)[0] for m in methods}
    rows = {m: [song_scores(s, embed, m, params[m], chunk, level) for s in test_songs] for m in methods}
    keys = list(dict.fromkeys(k for m in methods for k, v in rows[m][0].items() if isinstance(v, (int, float))))
    summary = {m: {k: _mean_ci([r.get(k, np.nan) for r in rows[m]]) for k in keys} for m in methods}
    tests = {}
    if reference in methods:
        for metric in ("completeness", "purity", "permutation_rate"):
            pv = []
            for m in methods:
                if m == reference:
                    continue
                c = paired_compare([r[metric] for r in rows[reference]], [r[metric] for r in rows[m]])
                tests[f"{reference} - {m} / {metric}"] = c
                pv.append((f"{reference} - {m} / {metric}", c.get("wilcoxon_p", np.nan)))
            for name, p_holm in _holm(pv).items():
                tests[name]["wilcoxon_p_holm"] = p_holm
    return {"level": level, "params": params, "n_test": len(test_songs), "summary": summary, "tests": tests,
            "songs": [s.name for s in test_songs], "rows": rows}


def _mean_ci(x: list[float], n_boot: int = 5000, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "ci95": [np.nan, np.nan], "n": 0}
    boots = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(v) for v in np.percentile(boots, [2.5, 97.5])], "n": int(len(x))}


def _holm(pvals: list[tuple[str, float]]) -> dict[str, float]:
    ok = [(n, p) for n, p in pvals if np.isfinite(p)]
    order = sorted(ok, key=lambda t: t[1])
    out, run = {}, 0.0
    for i, (n, p) in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p))
        out[n] = run
    return out


def markdown(result: dict) -> str:
    cols = ["purity", "completeness", "local_completeness", "permutation_rate", "count_error", "entry_f1"]
    lines = [f"Level: {result['level']}, {result['n_test']} test songs, mean [95 % CI]", "",
             "| method | param | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for m, s in result["summary"].items():
        cells = [f"{s[c]['mean']:.3f} [{s[c]['ci95'][0]:.3f}, {s[c]['ci95'][1]:.3f}]" if c in s else "-" for c in cols]
        lines.append(f"| {m} | {result['params'][m]} | " + " | ".join(cells) + " |")
    if result["tests"]:
        lines += ["", "| comparison | n | mean diff [95 % CI] | d_z | Wilcoxon p (Holm) |", "|---|---|---|---|---|"]
        for name, c in result["tests"].items():
            if "ci95" in c:
                lines.append(f"| {name} | {c['n']} | {c['mean_diff']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] "
                             f"| {c['d_z']:.2f} | {c.get('wilcoxon_p_holm', c['wilcoxon_p']):.3g} |")
    return "\n".join(lines)
