"""Phase 1 / 2 comparison: every tracking method on the same songs, thresholds tuned on a dev set only.

Protocol: each method's threshold is chosen on the ``tune`` songs (best mean ``f``, the harmonic mean of
song-wide purity and completeness), then frozen and scored on the ``test`` songs, which never influence
any choice. Each test song gives one paired observation per method; methods are compared to a reference
method with a bootstrap CI over songs, the paired effect size d_z and a Wilcoxon test (Holm-corrected
over the comparisons).

The tune songs are scored for every (method, threshold) of the grid, in parallel (one process per song,
``jobs``); the test songs then only at each method's chosen threshold (``full_grid`` scores them at every
threshold too, for diagnostics: same results, about ten times the cost). Per song, what does not depend on
the threshold (the k-means and linkage of the clusterings, the references' ownership) is computed once
and cut at every threshold (``track.*_sweep``), with labels identical to one run per threshold.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .features import Cached, load_cache
from .metrics import ownership, paired_compare, score
from .track import (online_registry, per_chunk, per_chunk_linked, per_chunk_linked_sweep, per_chunk_sweep, whole_song,
                    whole_song_sweep)


def _log(*a) -> None:  # flushed: a redirected log shows progress as it happens
    print(*a, flush=True)

METHODS = ("one", "per_chunk", "per_chunk_linked", "whole_song", "online", "online_two_pass")
# Phase 1 picked the edge of the first grids (0.8 for the clusterings, 0.2 for the registry): widened.
# MoisesDB (about 11 tracks per song) picked the lowest cut and the highest tau: widened again, both ways.
CUTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
GRID = {"one": [0.0], "per_chunk": CUTS, "per_chunk_linked": CUTS, "whole_song": CUTS,
        # online: tau, the cosine to join a known source (its new clusters are cut at 1 - tau)
        "online": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "online_two_pass": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}


def run_method(name: str, emb: np.ndarray, audible: np.ndarray, chunk: int, param: float, seed: int = 0
               ) -> tuple[np.ndarray, list | None]:
    if name == "one":  # trivial baseline: every audible token in one cluster
        return np.where(audible, 0, -1), None
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


def method_sweep(name: str, emb: np.ndarray, audible: np.ndarray, chunk: int, params, seed: int = 0,
                 device: str = "cpu") -> dict[float, tuple[np.ndarray, list | None]]:
    """{param: ``run_method(name, emb, audible, chunk, param, seed)``} for every param, sharing the work that
    does not depend on it (``track.*_sweep``: identical labels). The online registries depend on tau from
    their first chunk on, so they run once per tau. ``device`` "cuda": the micro k-means and the final
    assignment on the GPU (``track_gpu``; same algorithm, float32 rounding of another library)."""
    kw = {}
    if device != "cpu":
        from .track_gpu import gpu_ops

        kw = gpu_ops(device)
    if name == "per_chunk":
        return {g: (lab, None) for g, lab in
                per_chunk_sweep(emb, audible, chunk, params, seed, kmeans=kw.get("kmeans")).items()}
    if name == "per_chunk_linked":
        return {g: (lab, None) for g, lab in
                per_chunk_linked_sweep(emb, audible, chunk, params, seed, kmeans=kw.get("kmeans")).items()}
    if name == "whole_song":
        return {g: (lab, None) for g, lab in whole_song_sweep(emb, audible, params, seed, **kw).items()}
    return {g: run_method(name, emb, audible, chunk, g, seed) for g in params}


def _song_grid(args) -> tuple[str, dict]:
    """One song, every (method, param) of ``params`` [(method, params)]: {(method, param): scores}. Runs in a
    worker process. ``sweep`` False: the reference path, one ``run_method`` and ``ownership`` per cell."""
    song, head, chunk, level, params, sweep, device = args
    if isinstance(song, str):
        song = load_cache(song)
    embed = _embedder(head)
    emb = embed(song.tokens)  # once per song, shared by every method
    if not sweep:
        return song.name, {(m, g): song_scores(song, lambda _: emb, m, g, chunk, level) for m, gs in params for g in gs}
    groups = song.instruments if level == "instrument" else None
    energy = song.energy  # float16 -> float32 expansion once, not once per cell
    owned = ownership(energy, groups)
    out = {}
    for m, gs in params:
        for g, (lab, births) in method_sweep(m, emb, owned[1], chunk, gs, device=device).items():
            out[(m, g)] = score(lab, energy, chunk, groups, births=births, owned=owned)
    return song.name, out


def score_grid(songs: list[Cached], chunk: int, level: str, head: str | None = None, methods=METHODS,
               jobs: int = 1, log=_log, params: dict | None = None, sweep: bool = True, device: str = "cpu"
               ) -> dict[str, dict]:
    """{song name: {(method, param): scores}} for every song, at ``params`` {method: [param]} (default the
    whole ``GRID``); songs with a ``path`` are reloaded in the workers instead of being pickled across, so a
    worker holds one song at a time."""
    params = params or {m: GRID[m] for m in methods}
    todo = tuple((m, tuple(params[m])) for m in methods)
    tasks = [(s.path or s, head, chunk, level, todo, sweep, device) for s in songs]
    out = {}
    if jobs <= 1:
        for i, t in enumerate(tasks, 1):
            name, grid = _song_grid(t)
            out[name] = grid
            log(f"scored {i}/{len(tasks)}: {name}")
        return out
    # CUDA cannot be used in a forked child of a process that touched it: GPU workers start fresh.
    ctx = multiprocessing.get_context("spawn") if device != "cpu" else None
    with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
        for i, (name, grid) in enumerate(ex.map(_song_grid, tasks), 1):
            out[name] = grid
            log(f"scored {i}/{len(tasks)}: {name}")
    return out


def _f(rows: list[dict]) -> float:
    """Tuning objective: mean V-measure (purity / completeness would reward merging everything)."""
    return float(np.nanmean([r["v_measure"] for r in rows]))


def compare(tune_songs: list[Cached], test_songs: list[Cached], chunk: int, level: str = "track",
            methods=METHODS, reference: str = "online_two_pass", head: str | None = None, jobs: int = 1,
            log=_log, full_grid: bool = False, sweep: bool = True, device: str = "cpu") -> dict:
    """Tune each method on ``tune_songs``, score on ``test_songs`` -> per-song rows, means and paired tests.

    ``full_grid``: also score the test songs at every threshold (diagnostics, the old behaviour; the results
    are the same). Otherwise their ``grid`` entries hold only the chosen (method, param) cells.
    ``sweep`` False: the per-threshold reference path (slow, for equivalence checks)."""
    tune_grid = score_grid(tune_songs, chunk, level, head, methods, jobs, log, sweep=sweep, device=device)
    params, tuning = {}, {}
    for m in methods:
        res = {g: _f([tune_grid[s.name][(m, g)] for s in tune_songs]) for g in GRID[m]}
        params[m], tuning[m] = max(res, key=res.get), res
        log(f"tune {m} ({level}): " + ", ".join(f"{g}: {v:.3f}" for g, v in res.items()) + f" -> {params[m]}")
    test_grid = score_grid(test_songs, chunk, level, head, methods, jobs, log,
                           params=None if full_grid else {m: [params[m]] for m in methods}, sweep=sweep, device=device)
    rows = {m: [test_grid[s.name][(m, params[m])] for s in test_songs] for m in methods}
    keys = list(dict.fromkeys(k for m in methods for k, v in rows[m][0].items() if isinstance(v, (int, float))))
    summary = {m: {k: _mean_ci([r.get(k, np.nan) for r in rows[m]]) for k in keys} for m in methods}
    tests = {}
    if reference in methods:
        pv = []
        for metric in ("v_measure", "completeness", "purity", "permutation_rate"):
            for m in methods:
                if m == reference:
                    continue
                name = f"{reference} - {m} / {metric}"
                tests[name] = paired_compare([r[metric] for r in rows[reference]], [r[metric] for r in rows[m]])
                pv.append((name, tests[name].get("wilcoxon_p", np.nan)))
        for name, p_holm in _holm(pv).items():
            tests[name]["wilcoxon_p_holm"] = p_holm
    both = {name: {**tune_grid.get(name, {}), **test_grid.get(name, {})} for name in {**tune_grid, **test_grid}}
    grid = {name: {f"{m}@{g}": r for (m, g), r in g_rows.items()} for name, g_rows in both.items()}
    return {"level": level, "params": params, "tuning": tuning, "n_tune": len(tune_songs),
            "n_test": len(test_songs), "summary": summary, "tests": tests, "songs": [s.name for s in test_songs],
            "rows": rows, "grid": grid}


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
    cols = ["v_measure", "local_v_measure", "homogeneity", "v_completeness", "permutation_rate", "n_clusters", "entry_f1"]
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


MERGE_CUTS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def _song_merge(args) -> tuple[str, dict]:
    """One song: the method at its tuned threshold, then co-activation merges (every cut, and oracle n)."""
    from .track import merge_coactive

    path, chunk, method, param = args
    song = load_cache(path)
    _, audible = ownership(song.energy)
    lab, _ = run_method(method, song.tokens, audible, chunk, param)
    out = {"none": score(lab, song.energy, chunk)}
    for c in MERGE_CUTS:
        out[str(c)] = score(merge_coactive(lab, threshold=c), song.energy, chunk)
    out["oracle_n"] = score(merge_coactive(lab, n=song.energy.shape[0]), song.energy, chunk)
    return song.name, out


def merge_compare(tune_songs: list[Cached], test_songs: list[Cached], chunk: int, params: dict[str, float],
                  jobs: int = 1, log=_log) -> dict:
    """Co-activation merge on top of each method's tuned labels: merge cut tuned on ``tune_songs`` (V-measure),
    scored on ``test_songs``, next to no merge and the oracle-n merge (an upper bound, n = true stem count)."""
    out = {}
    for method, param in params.items():
        if method == "one":
            continue
        tasks = [(s.path, chunk, method, param) for s in tune_songs + test_songs]
        with ProcessPoolExecutor(jobs) as ex:
            rows = dict(ex.map(_song_merge, tasks))
        cut = max(MERGE_CUTS, key=lambda c: np.mean([rows[s.name][str(c)]["v_measure"] for s in tune_songs]))
        test = {k: [rows[s.name][k]["v_measure"] for s in test_songs] for k in ("none", str(cut), "oracle_n")}
        out[method] = {"param": param, "merge_cut": cut, "songs": [s.name for s in test_songs],
                       "v_none": test["none"], "v_merged": test[str(cut)], "v_oracle_n": test["oracle_n"],
                       "clusters_merged": [rows[s.name][str(cut)]["n_clusters"] for s in test_songs]}
        log(f"{method}: V none {np.mean(test['none']):.3f}, merged (cut {cut}) {np.mean(test[str(cut)]):.3f}, "
            f"oracle-n {np.mean(test['oracle_n']):.3f}")
    return out
