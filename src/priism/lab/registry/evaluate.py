"""Phase 1 / 2 comparison: every tracking method on the same songs, thresholds tuned on a dev set only.

Protocol: each method's threshold is chosen on the ``tune`` songs (best mean ``f``, the harmonic mean of
song-wide purity and completeness), then frozen and scored on the ``test`` songs, which never influence
any choice. Each test song gives one paired observation per method; methods are compared to a reference
method with a bootstrap CI over songs, the paired effect size d_z and a Wilcoxon test (Holm-corrected
over the comparisons).

Every song is scored once for every (method, threshold) of the grid, in parallel (one process per song,
``jobs``): tuning then only reads the table, and the test songs' rows at the chosen thresholds are the
results. Scoring a test song at thresholds it was not tuned for costs time, not validity.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .features import Cached, load_cache
from .metrics import ownership, paired_compare, score
from .track import online_registry, per_chunk, per_chunk_linked, whole_song

METHODS = ("per_chunk", "per_chunk_linked", "whole_song", "online", "online_two_pass")
# Phase 1 picked the edge of the first grids (0.8 for the clusterings, 0.2 for the registry): widened.
CUTS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
GRID = {"per_chunk": CUTS, "per_chunk_linked": CUTS, "whole_song": CUTS,
        # online: tau, the cosine to join a known source (its new clusters are cut at 1 - tau)
        "online": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "online_two_pass": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}


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


def song_scores(song: Cached, embed, method: str, param: float, chunk: int, level: str, seed: int = 0) -> dict:
    groups = song.instruments if level == "instrument" else None
    _, audible = ownership(song.energy, groups)
    lab, births = run_method(method, embed(song.tokens), audible, chunk, param, seed)
    return score(lab, song.energy, chunk, groups, births=births)


_HEADS: dict = {}


def _embedder(head: str | None):
    if head is None:
        return lambda x: x
    if head not in _HEADS:
        from .head import load_head

        _HEADS[head] = load_head(head)
    return _HEADS[head].embed


def _song_grid(args) -> tuple[str, dict]:
    """One song, every (method, threshold): {(method, param): scores}. Runs in a worker process."""
    song, head, chunk, level, methods = args
    if isinstance(song, str):
        song = load_cache(song)
    embed = _embedder(head)
    emb = embed(song.tokens)  # once per song, shared by every method
    return song.name, {(m, g): song_scores(song, lambda _: emb, m, g, chunk, level) for m in methods for g in GRID[m]}


def score_grid(songs: list[Cached], chunk: int, level: str, head: str | None = None, methods=METHODS,
               jobs: int = 1, log=print) -> dict[str, dict]:
    """{song name: {(method, param): scores}} for every song; songs with a ``path`` are reloaded in the
    workers instead of being pickled across."""
    tasks = [(s.path or s, head, chunk, level, tuple(methods)) for s in songs]
    out = {}
    if jobs <= 1:
        for i, t in enumerate(tasks, 1):
            name, grid = _song_grid(t)
            out[name] = grid
            log(f"scored {i}/{len(tasks)}: {name}")
        return out
    with ProcessPoolExecutor(jobs) as ex:
        for i, (name, grid) in enumerate(ex.map(_song_grid, tasks), 1):
            out[name] = grid
            log(f"scored {i}/{len(tasks)}: {name}")
    return out


def _f(rows: list[dict]) -> float:
    p, c = np.nanmean([r["purity"] for r in rows]), np.nanmean([r["completeness"] for r in rows])
    return float(2 * p * c / (p + c)) if p + c > 0 else 0.0


def compare(tune_songs: list[Cached], test_songs: list[Cached], chunk: int, level: str = "track",
            methods=METHODS, reference: str = "online_two_pass", head: str | None = None, jobs: int = 1,
            log=print) -> dict:
    """Tune each method on ``tune_songs``, score on ``test_songs`` -> per-song rows, means and paired tests."""
    tune_grid = score_grid(tune_songs, chunk, level, head, methods, jobs, log)
    test_grid = score_grid(test_songs, chunk, level, head, methods, jobs, log)
    params, tuning = {}, {}
    for m in methods:
        res = {g: _f([tune_grid[s.name][(m, g)] for s in tune_songs]) for g in GRID[m]}
        params[m], tuning[m] = max(res, key=res.get), res
        log(f"tune {m} ({level}): " + ", ".join(f"{g}: {v:.3f}" for g, v in res.items()) + f" -> {params[m]}")
    rows = {m: [test_grid[s.name][(m, params[m])] for s in test_songs] for m in methods}
    keys = list(dict.fromkeys(k for m in methods for k, v in rows[m][0].items() if isinstance(v, (int, float))))
    summary = {m: {k: _mean_ci([r.get(k, np.nan) for r in rows[m]]) for k in keys} for m in methods}
    tests = {}
    if reference in methods:
        pv = []
        for metric in ("completeness", "purity", "permutation_rate"):
            for m in methods:
                if m == reference:
                    continue
                name = f"{reference} - {m} / {metric}"
                tests[name] = paired_compare([r[metric] for r in rows[reference]], [r[metric] for r in rows[m]])
                pv.append((name, tests[name].get("wilcoxon_p", np.nan)))
        for name, p_holm in _holm(pv).items():
            tests[name]["wilcoxon_p_holm"] = p_holm
    return {"level": level, "params": params, "tuning": tuning, "n_tune": len(tune_songs),
            "n_test": len(test_songs), "summary": summary, "tests": tests, "songs": [s.name for s in test_songs],
            "rows": rows}


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
    lines = [f"Level: {result['level']}, tuned on {result['n_tune']} songs, {result['n_test']} test songs, "
             "mean [95 % CI]", "",
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
