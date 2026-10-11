"""The agent loop on the GPU machine (``priism-lab agent``): one job at a time, watched, synced.

Each pass: heartbeat, finish syncing any job that ended while offline, then take the oldest pending job.
A GPU job starts only when the GPU is free (low utilisation and memory over a few samples): the machine
is someone's PC, and his own use comes first.

While a job runs, every ``tick`` seconds the agent checks, following the lab rule that no long run is left
unwatched (same checks as ``pod/guard.py``):
- a ``cancel/<id>`` marker in the store: stop;
- no new log output for ``stall_minutes``: stop;
- a GPU job whose GPU stays at 0 % for ``gpu_idle_minutes``: stop;
- ``max_minutes`` reached: stop.
Every ``sync_every`` seconds it uploads the log and a status (heartbeat, last line). A failed upload only
marks the agent offline: the job goes on, and the outcome is kept in ``.agent/finished/`` until it has
been uploaded (results included), so a lost connection loses nothing.

The job runs in its own process session. If the agent itself restarts (crash, reboot of the service), it
finds the job in ``.agent/current.json``: still alive, it is watched again; gone, it is reported lost.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .jobs import Job, JobRejected, commands
from .store import StoreError


def gpu_stats() -> tuple[int, int] | None:
    """(utilisation %, memory used MiB) of the busiest GPU, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        rows = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines()]
        return max(r[0] for r in rows), max(r[1] for r in rows)
    except Exception:
        return None


class _Pid:
    """A job process found again after an agent restart: no Popen, only its pid."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            return -1  # gone; its exit code is unknown


def _kill(proc) -> None:
    """SIGTERM to the job's whole process group, SIGKILL 30 s later if it is still there."""
    if hasattr(os, "killpg"):
        try:
            pg = os.getpgid(proc.pid)
        except OSError:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pg, sig)
            except OSError:
                return
            for _ in range(30):
                if proc.poll() is not None:
                    return
                time.sleep(1)
    else:  # Windows (tests only)
        proc.terminate()


class Agent:
    def __init__(self, store, workdir: str | Path, repo: str | Path, name: str | None = None,
                 bin_dir: str | Path | None = None, gpu=gpu_stats, tick: float = 10.0, sync_every: float = 30.0,
                 free_util: int = 10, free_mem_mib: int = 1500, free_samples: int = 3, free_gap_s: float = 5.0,
                 log=print):
        self.store, self.workdir, self.repo = store, Path(workdir), Path(repo)
        self.name = name or socket.gethostname()
        self.bin_dir = Path(bin_dir or Path(sys.executable).parent)
        self.gpu, self.tick, self.sync_every, self.log = gpu, tick, sync_every, log
        self.free = (free_util, free_mem_mib, free_samples, free_gap_s)
        self.state_dir = self.workdir / ".agent"
        (self.state_dir / "finished").mkdir(parents=True, exist_ok=True)
        (self.workdir / "logs").mkdir(exist_ok=True)
        self.online = True
        self.restart = False

    # --------------------------------------------------------------------------------------- store access

    def _try(self, fn, *a) -> bool:
        try:
            fn(*a)
            if not self.online:
                self.log("store reachable again")
            self.online = True
            return True
        except (StoreError, OSError) as e:
            if self.online:
                self.log(f"store unreachable ({e}); the job goes on, syncing later")
            self.online = False
            return False

    def heartbeat(self, state: str, job: str | None = None) -> None:
        g = self.gpu()
        self._try(self.store.put, f"agents/{self.name}.json", json.dumps(
            {"time": time.time(), "state": state, "job": job, "gpu": g, "version": self._version()}).encode())

    def _version(self) -> str:
        try:
            return subprocess.run(["git", "-C", str(self.repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "?"

    # ---------------------------------------------------------------------------------------------- loop

    def run_forever(self, idle_poll: float = 60.0) -> None:
        self.log(f"agent {self.name} on {self.workdir}")
        while True:
            try:
                ran = self.step()
            except Exception as e:  # the agent must never die on one bad pass
                self.log(f"agent error: {e!r}")
                ran = False
            if self.restart:
                self.log("updated: exiting so the service restarts the agent on the new code")
                return
            if not ran:
                time.sleep(idle_poll)

    def step(self) -> bool:
        """One pass; True if a job ran (look again at once), False if idle or waiting."""
        self._resume()
        self._flush()
        try:
            pending = self.store.list("queue/pending/")
        except (StoreError, OSError):
            self.online = False
            return False
        if not pending:
            self.heartbeat("idle")
            return False
        key = pending[0]
        job = Job.from_json(self.store.get(key) or b"{}")
        job.id = job.id or Path(key).stem
        try:
            cmds = commands(job, self.repo, self.bin_dir)
        except (JobRejected, TypeError) as e:
            self.log(f"rejected {job.id}: {e}")
            self.store.put(f"queue/failed/{job.id}.json", job.to_json())
            self.store.put(f"status/{job.id}.json", json.dumps({"state": "failed", "reason": f"rejected: {e}",
                                                                 "time": time.time()}).encode())
            self.store.delete(key)
            return True
        if job.gpu and not self.gpu_free():
            self.heartbeat("waiting: GPU busy", job.id)
            return False
        self.store.put(f"queue/running/{job.id}.json", job.to_json())
        self.store.delete(key)
        self.run_job(job, cmds)
        return True

    def gpu_free(self) -> bool:
        util, mem, n, gap = self.free
        for i in range(n):
            g = self.gpu()
            if g is None:
                return True  # no nvidia-smi: nothing to measure; a GPU job will fail on its own
            if g[0] > util or g[1] > mem:
                return False
            if i < n - 1:
                time.sleep(gap)
        return True

    # ------------------------------------------------------------------------------------------- one job

    def run_job(self, job: Job, cmds: list[list[str]]) -> None:
        logf = self.workdir / "logs" / f"{job.id}.log"
        start = time.time()
        env = {**os.environ, **job.env, "PATH": f"{self.bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
               "PYTHONUNBUFFERED": "1", "PRIISM": str(self.repo)}
        exit_code, reason = 0, "ok"
        with logf.open("ab") as out:
            for cmd in cmds:
                out.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n".encode())
                out.flush()
                kw = {"start_new_session": True} if os.name != "nt" else {}
                proc = subprocess.Popen(cmd, cwd=self.workdir, stdout=out, stderr=subprocess.STDOUT, env=env, **kw)
                self._save_current(job, proc.pid, start)
                exit_code, reason = self.watch(job, proc, logf, start)
                if exit_code != 0 or reason != "ok":
                    break
        self._finish(job, exit_code, reason, start)
        if job.kind == "update" and reason == "ok":
            self.restart = True  # new code is installed: run_forever exits, the service starts it again

    def watch(self, job: Job, proc, logf: Path, start: float) -> tuple[int, str]:
        idle_since, last_sync = None, 0.0
        while True:
            code = proc.poll()
            if code is not None:
                return code, "ok" if code == 0 else f"exit code {code}"
            now = time.time()
            stop = None
            if now - max(logf.stat().st_mtime, start) > job.stall_minutes * 60:
                stop = f"no log output for {job.stall_minutes:g} min"
            elif now - start > job.max_minutes * 60:
                stop = f"time cap {job.max_minutes:g} min"
            elif job.gpu:
                g = self.gpu()
                if g is not None and g[0] == 0:
                    idle_since = idle_since or now
                    if now - idle_since > job.gpu_idle_minutes * 60:
                        stop = f"GPU idle for {job.gpu_idle_minutes:g} min"
                else:
                    idle_since = None
            if stop is None and now - last_sync >= self.sync_every:
                last_sync = now
                self._sync(job, logf, start, "running")
                try:
                    if self.store.get(f"cancel/{job.id}") is not None:
                        stop = "cancelled"
                except (StoreError, OSError):
                    self.online = False
            if stop:
                with logf.open("a") as f:
                    f.write(f"### agent {time.strftime('%Y-%m-%d %H:%M:%S')}: {stop}, stopping\n")
                self.log(f"{job.id}: {stop}")
                _kill(proc)
                return -1, stop
            time.sleep(self.tick)

    def _sync(self, job: Job, logf: Path, start: float, state: str, extra: dict | None = None) -> bool:
        data = logf.read_bytes()[-2_000_000:] if logf.exists() else b""
        lines = data.decode(errors="replace").strip().splitlines()
        st = {"state": state, "time": time.time(), "minutes": (time.time() - start) / 60,
              "last_line": lines[-1][-300:] if lines else "", "agent": self.name, **(extra or {})}
        ok = self._try(self.store.put, f"logs/{job.id}.log", data)
        ok = ok and self._try(self.store.put, f"status/{job.id}.json", json.dumps(st).encode())
        if state == "running":
            self.heartbeat("running", job.id)
        return ok

    # ------------------------------------------------------------------------- local state, offline-safe

    def _save_current(self, job: Job, pid: int, start: float) -> None:
        (self.state_dir / "current.json").write_text(json.dumps({"job": json.loads(job.to_json()), "pid": pid,
                                                                  "start": start}))

    def _finish(self, job: Job, exit_code: int, reason: str, start: float) -> None:
        rec = {"job": json.loads(job.to_json()), "exit_code": exit_code, "reason": reason, "start": start,
               "end": time.time()}
        (self.state_dir / "finished" / f"{job.id}.json").write_text(json.dumps(rec))
        (self.state_dir / "current.json").unlink(missing_ok=True)
        self.log(f"{job.id}: {reason}")
        self._flush()

    def _flush(self) -> None:
        """Upload every finished job's log, results and outcome; keep the record until all of it went up."""
        for f in sorted((self.state_dir / "finished").glob("*.json")):
            rec = json.loads(f.read_text())
            job = Job(**rec["job"])
            ok = self._sync(job, self.workdir / "logs" / f"{job.id}.log", rec["start"],
                            "done" if rec["exit_code"] == 0 and rec["reason"] == "ok" else "failed",
                            {"exit_code": rec["exit_code"], "reason": rec["reason"], "synced": False})
            if not ok:
                return
            for folder in job.results:
                for p in sorted((self.workdir / folder).rglob("*")) if (self.workdir / folder).exists() else []:
                    if p.is_file() and p.stat().st_mtime >= rec["start"] - 5:
                        key = f"results/{job.id}/{p.relative_to(self.workdir).as_posix()}"
                        if not self._try(self.store.upload_file, key, p):
                            return
            state = "done" if rec["exit_code"] == 0 and rec["reason"] == "ok" else "failed"
            if not (self._try(self.store.put, f"queue/{state}/{job.id}.json", job.to_json())
                    and self._try(self.store.delete, f"queue/running/{job.id}.json")
                    and self._try(self.store.delete, f"cancel/{job.id}")
                    and self._sync(job, self.workdir / "logs" / f"{job.id}.log", rec["start"], state,
                                   {"exit_code": rec["exit_code"], "reason": rec["reason"], "synced": True})):
                return
            f.unlink()

    def _resume(self) -> None:
        cur = self.state_dir / "current.json"
        if not cur.exists():
            return
        d = json.loads(cur.read_text())
        job = Job(**d["job"])
        proc = _Pid(d["pid"])
        logf = self.workdir / "logs" / f"{job.id}.log"
        if proc.poll() is None:
            self.log(f"{job.id}: still running after an agent restart, watching it again")
            code, reason = self.watch(job, proc, logf, d["start"])
            if reason == "exit code -1":  # a _Pid cannot read the exit code: count it as failed, say why
                reason = "ended, exit code unknown (the agent had restarted): check the log"
            self._finish(job, code, reason, d["start"])
        else:
            self._finish(job, -1, "lost: the agent restarted and the job had ended (exit code unknown)", d["start"])
