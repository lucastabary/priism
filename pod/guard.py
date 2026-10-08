"""Queue guard: stops a running job that is stuck, so the worker moves on to the next one.

Runs next to the worker on the pod (pod/setup.sh starts it), independent of any remote session. Every
minute it looks at each job in <queue>/running:
- no new log output for ``# stall: N`` minutes (default 20 for the gpu lane, 60 for the cpu lane);
- or, for a gpu-lane job, the GPU at 0 % for ``--gpu-idle`` minutes in a row.
Such a job's process group gets SIGTERM, then SIGKILL; the worker sees it end, files it under failed,
and starts the next pending job. Each stop is written to the job's log and to <queue>/guard.log.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path

_STALL_RE = re.compile(r"^#\s*stall:\s*(\d+)", re.M)
_LANE_RE = re.compile(r"^#\s*lane:\s*(\w+)", re.M)


def gpu_util() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return max(int(x) for x in out.split())
    except Exception:  # no GPU tool or unreadable: never a reason to stop a job
        return None


def job_pgids(script: Path) -> set[int]:
    """Process groups of the bash processes running ``script`` (the worker starts each in its own session)."""
    groups = set()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if len(cmd) >= 2 and cmd[0].endswith(b"bash") and cmd[1].decode(errors="replace").endswith(
                f"running/{script.name}"):
            try:
                groups.add(os.getpgid(int(d.name)))
            except OSError:
                pass
    return groups


def stop(queue: Path, script: Path, reason: str, log=print) -> None:
    msg = f"### guard {time.strftime('%Y-%m-%d %H:%M:%S')}: {reason}, stopping {script.name}"
    log(msg)
    with (queue / "logs" / script.name.replace(".sh", ".log")).open("a") as f:
        f.write(msg + "\n")
    with (queue / "guard.log").open("a") as f:
        f.write(msg + "\n")
    groups = job_pgids(script)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for g in groups:
            try:
                os.killpg(g, sig)
            except OSError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(30)


def check(queue: Path, idle_since: dict, gpu_idle_min: float, now: float | None = None, util=gpu_util,
          stopper=stop) -> list[str]:
    """One pass; returns the jobs it stopped. ``idle_since`` keeps when the GPU went idle per job."""
    now = time.time() if now is None else now
    stopped = []
    u = None
    for script in sorted((queue / "running").glob("*.sh")):
        text = script.read_text(errors="replace")
        m = _LANE_RE.search(text)
        lane = m.group(1) if m else "gpu"
        m = _STALL_RE.search(text)
        limit = float(m.group(1)) if m else (20.0 if lane == "gpu" else 60.0)
        logf = queue / "logs" / script.name.replace(".sh", ".log")
        last = max(logf.stat().st_mtime if logf.exists() else 0.0, script.stat().st_mtime)
        if now - last > limit * 60:
            stopper(queue, script, f"no log output for {limit:g} min")
            stopped.append(script.name)
            continue
        if lane == "gpu":
            u = util() if u is None else u
            if u == 0:
                idle_since.setdefault(script.name, now)
                if now - idle_since[script.name] > gpu_idle_min * 60:
                    stopper(queue, script, f"GPU idle for {gpu_idle_min:g} min")
                    stopped.append(script.name)
                    idle_since.pop(script.name, None)
            else:
                idle_since.pop(script.name, None)
    return stopped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queue", default=os.environ.get("PRIISM_QUEUE", "/workspace/priism/queue"))
    p.add_argument("--gpu-idle", type=float, default=15.0, help="minutes of 0 %% GPU before a gpu job is stopped")
    p.add_argument("--every", type=float, default=60.0, help="seconds between checks")
    a = p.parse_args()
    queue, idle = Path(a.queue), {}
    print(f"guard on {queue}", flush=True)
    while True:
        try:
            check(queue, idle, a.gpu_idle)
        except Exception as e:  # the guard itself must never die
            print(f"guard error: {e}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
