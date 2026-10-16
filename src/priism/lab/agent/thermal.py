"""GPU temperature and power: the agent's samples, quiet mode's temperature controller, and a measurement load.

The agent samples the GPU every ``thermal_every`` seconds while a job runs (a thread of its own, so frozen
parts of quiet mode are measured too) and writes each sample to ``.agent/thermal.jsonl`` and to the job's
``logs/<id>.thermal.jsonl``, uploaded as ``results/<id>/thermal.jsonl``. ``gpu-load`` (a lab command an
agent job may run) keeps the GPU busy for a set time, so a job with ``--quiet off`` then one with
``--quiet on`` give the machine's real curves; ``thermal-report`` summarises the pulled files.

Nothing here imports torch at module level: the agent must stay light.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

FIELDS = ("temp_c", "power_w", "util")


def gpu_thermal() -> dict | None:
    """Temperature (C), power draw (W) and utilisation (%) of the first GPU, or None without nvidia-smi.
    A field the driver reports as N/A (power on some laptop GPUs) is None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        line = out.strip().splitlines()[0]
    except Exception:
        return None
    vals = []
    for v in line.split(","):
        try:
            vals.append(float(v))
        except ValueError:  # "[N/A]", "[Not Supported]"
            vals.append(None)
    if len(vals) != len(FIELDS) or all(v is None for v in vals):
        return None
    return dict(zip(FIELDS, vals))


def next_pause(pause: float, temp: float, target: float, band: float, lo: float, hi: float) -> float:
    """One step of quiet mode's temperature controller, on the frozen part of each cycle (the run bursts stay
    fixed): above the target the pauses grow by half, more than ``band`` below it they shrink by a fifth, in
    between they hold (the hysteresis that keeps it from oscillating around the target). Faster up than down,
    since being too hot is what the owner hears. Kept within [lo, hi]."""
    if temp > target:
        pause *= 1.5
    elif temp < target - band:
        pause /= 1.25
    return min(max(lo, hi), max(lo, pause))


def summarize(samples: list[dict]) -> dict:
    """Means of the readable fields over samples (None when a field was never readable)."""
    out = {}
    for f in FIELDS:
        vals = [s[f] for s in samples if s.get(f) is not None]
        out[f] = round(sum(vals) / len(vals), 1) if vals else None
    return out


def append_jsonl(path: Path, rec: dict, cap_bytes: int | None = None) -> None:
    """Append one line; past ``cap_bytes`` the file is rotated to ``<name>.1`` (one old file kept)."""
    if cap_bytes is not None and path.exists() and path.stat().st_size > cap_bytes:
        path.replace(path.with_name(path.name + ".1"))
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass  # a line cut by a kill
    return out


def report(paths: list[str], last_minutes: float = 5.0) -> list[dict]:
    """One summary per thermal.jsonl: duration, means and max over the whole run and over its last
    ``last_minutes`` (the steady state, once the cooler has caught up), share of samples in quiet mode."""
    rows = []
    for p in paths:
        s = read_jsonl(p)
        if not s:
            rows.append({"file": str(p), "samples": 0})
            continue
        t0, t1 = s[0]["time"], s[-1]["time"]
        tail = [x for x in s if x["time"] >= t1 - last_minutes * 60]
        temps = [x["temp_c"] for x in s if x.get("temp_c") is not None]
        pauses = [x["pause_s"] for x in s if x.get("quiet")]
        rows.append({"file": str(p), "samples": len(s), "minutes": round((t1 - t0) / 60, 1),
                     **{f"mean_{k}": v for k, v in summarize(s).items()},
                     "max_temp_c": max(temps) if temps else None,
                     **{f"last{last_minutes:g}min_{k}": v for k, v in summarize(tail).items()},
                     "quiet_share": round(sum(bool(x.get("quiet")) for x in s) / len(s), 2),
                     "mean_pause_s": round(sum(pauses) / len(pauses), 1) if pauses else None})
    return rows


def gpu_load(minutes: float, size: int = 4096, device: str = "cuda", report_every: float = 10.0) -> dict:
    """Matrix products back to back for ``minutes`` of wall time (frozen time included, so a quiet run and a
    normal one last as long), one JSON line every ``report_every`` s: elapsed time, products, TFLOP/s since
    the previous line. The throughput ratio of a quiet run to a normal one is the duty cycle the job got."""
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("gpu-load: no CUDA device (use --device cpu to try it)")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    flop = 2 * size ** 3
    t0 = last = time.time()
    n = n_last = 0
    while time.time() - t0 < minutes * 60:
        a = (a @ b).clamp_(-1, 1)  # clamp: keeps fp16 finite over thousands of products
        n += 1
        if n % 8 == 0 and device.startswith("cuda"):
            torch.cuda.synchronize()
        now = time.time()
        if now - last >= report_every:
            print(json.dumps({"elapsed_s": round(now - t0, 1), "products": n,
                              "tflops": round((n - n_last) * flop / (now - last) / 1e12, 2)}), flush=True)
            last, n_last = now, n
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    out = {"minutes": round(dt / 60, 2), "products": n, "mean_tflops": round(n * flop / dt / 1e12, 2),
           "size": size, "device": device}
    print(json.dumps(out), flush=True)
    return out
