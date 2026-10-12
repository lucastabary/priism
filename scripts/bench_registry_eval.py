"""Benchmark and equivalence check of ``priism-lab registry-eval``'s fast path against the reference path.

Runs ``evaluate.compare`` on the same cached songs three ways and prints wall times and differences:

- ``old``: the reference, as before the fast path: every song (tune and test) at every (method, threshold),
  one clustering and one ownership per cell, on the CPU (``full_grid=True, sweep=False``);
- ``new``: threshold sweeps and tune-then-test on the CPU: must be identical to ``old`` (exactly);
- ``new-<device>`` (with ``--device cuda``): the same with the k-means and assignments on the GPU: float32
  rounding of another library, so small differences are possible and reported, not asserted.

Then a per-stage profile of one song (ownership, each method's sweep, scoring), CPU vs device. Memory: peak
RSS of this process (``--jobs 1``, the default, runs everything in it) and peak CUDA memory.

    python scripts/bench_registry_eval.py --cache /workspace/registry/cache --tune moises%dev2 \\
        --test moises%test --n-tune 6 --n-test 10 --device cuda

Exit status 1 if ``new`` differs from ``old`` (a bug); GPU differences only print.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # as registry-eval: one BLAS thread per process

import numpy as np  # noqa: E402

from priism.lab.registry.evaluate import GRID, compare, method_sweep  # noqa: E402
from priism.lab.registry.features import load_cache  # noqa: E402
from priism.lab.registry.metrics import ownership, score  # noqa: E402
from priism.lab.registry.run import select  # noqa: E402


def peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    except ImportError:  # Windows
        try:
            import psutil

            return psutil.Process().memory_info().peak_wset / 2**20
        except Exception:
            return math.nan


def diff(a, b, path="", out=None) -> list[tuple[str, float]]:
    """[(path, |a - b|)] of every differing leaf (NaN == NaN; inf for a structural difference)."""
    out = [] if out is None else out
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", math.inf))
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append((path, math.inf))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        if not (math.isnan(a) and math.isnan(b)) and a != b:
            out.append((path, abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf))
    elif a != b:
        out.append((path, math.inf))
    return out


def report(name: str, ref: dict, got: dict, methods) -> int:
    """Print how ``got`` differs from ``ref`` (params, per-song test rows); returns the count of differences."""
    same_params = ref["params"] == got["params"]
    print(f"  {name}: chosen thresholds {'identical' if same_params else 'DIFFER'}: {got['params']}"
          + ("" if same_params else f" (reference {ref['params']})"))
    n = 0 if same_params else 1
    for key in ("tuning", "rows", "summary"):
        d = diff(ref[key], got[key])
        n += len(d)
        worst = max((v for _, v in d), default=0.0)
        print(f"  {name}: {key}: {len(d)} differing values, max |diff| {worst:.3g}")
    for m in methods:
        dv = [abs(a["v_measure"] - b["v_measure"]) for a, b in zip(ref["rows"][m], got["rows"][m])
              if not (math.isnan(a["v_measure"]) and math.isnan(b["v_measure"]))]
        bad = sum(v > 1e-9 for v in dv)
        print(f"    {m:18s} test songs with |dV| > 1e-9: {bad}/{len(dv)}, max |dV| {max(dv, default=0):.3g}")
    return n


def profile_song(path: str, methods, chunk: int, level: str, device: str) -> None:
    song = load_cache(path)
    t = time.perf_counter()
    groups = song.instruments if level == "instrument" else None
    energy = song.energy
    owned = ownership(energy, groups)
    print(f"  {song.name}: tokens {song.tokens.shape}, {energy.shape[0]} sources, {int(owned[1].sum())} audible "
          f"tokens; energy + ownership {time.perf_counter() - t:.2f} s")
    for dev in ["cpu"] + ([device] if device != "cpu" else []):
        for m in methods:
            t = time.perf_counter()
            labs = method_sweep(m, song.tokens, owned[1], chunk, GRID[m], device=dev)
            t1 = time.perf_counter()
            for g, (lab, births) in labs.items():
                score(lab, energy, chunk, groups, births=births, owned=owned)
            t2 = time.perf_counter()
            print(f"    {dev:5s} {m:18s} {len(GRID[m]):2d} thresholds: method {t1 - t:6.2f} s, scores {t2 - t1:5.2f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tune", required=True, help="set spec, e.g. moises%%dev2")
    ap.add_argument("--test", required=True, help="set spec, e.g. moises%%test")
    ap.add_argument("--n-tune", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--methods", default="one,whole_song,online")
    ap.add_argument("--level", default="track", choices=["track", "instrument"])
    ap.add_argument("--chunk", type=float, default=4.0)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or torch:cpu (the GPU code on the CPU)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-old", action="store_true", help="no reference run (it is the slow one)")
    args = ap.parse_args()

    methods = tuple(args.methods.split(","))
    tune = select(args.cache, args.tune, lazy=True)[: args.n_tune]
    test = select(args.cache, args.test, lazy=True)[: args.n_test]
    chunk = max(1, round(args.chunk / test[0].frame_s))
    print(f"{len(tune)} tune + {len(test)} test songs, methods {methods}, level {args.level}, chunk {chunk} frames, "
          f"jobs {args.jobs}, OMP threads {os.environ['OMP_NUM_THREADS']}")
    quiet = lambda *_: None  # noqa: E731
    runs, times = {}, {}
    paths = [("new", "cpu", True, False)] + ([("old", "cpu", False, True)] if not args.skip_old else [])
    if args.device != "cpu":
        paths.append((f"new-{args.device}", args.device, True, False))
    for name, dev, sweep, full in paths:
        t = time.perf_counter()
        runs[name] = compare(tune, test, chunk, args.level, methods, reference=methods[-1], jobs=args.jobs,
                             log=quiet, full_grid=full, sweep=sweep, device=dev)
        times[name] = time.perf_counter() - t
        print(f"{name:12s} {times[name]:8.1f} s  ({times[name] / (len(tune) + len(test)):.1f} s per song), "
              f"peak RSS so far {peak_rss_mb():.0f} MB")
    if "old" in times:
        print(f"speedup new vs old: {times['old'] / times['new']:.1f}x")
    if args.device != "cpu":
        print(f"speedup new-{args.device} vs new: {times['new'] / times[f'new-{args.device}']:.2f}x"
              + (f", vs old: {times['old'] / times[f'new-{args.device}']:.1f}x" if "old" in times else ""))
        if args.device.startswith("cuda"):
            import torch

            print(f"peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

    bad = 0
    print("equivalence:")
    if "old" in runs:
        bad = report("new vs old (must be exact)", runs["old"], runs["new"], methods)
    if args.device != "cpu":
        report(f"new-{args.device} vs new (float tolerance)", runs["new"], runs[f"new-{args.device}"], methods)

    print("per-stage profile of the first tune song:")
    profile_song(tune[0].path, methods, chunk, args.level, args.device)
    print("RESULT:", "OK" if bad == 0 else f"FAIL ({bad} differences between new and old)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
