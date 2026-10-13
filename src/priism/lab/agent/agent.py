"""The agent loop on the GPU machine (``priism-lab agent``): one job at a time, watched, synced.

Each pass: heartbeat, finish syncing any job that ended while offline, then take the oldest pending job.
A job starts whatever the GPU is doing: the owner decides, with the tray's pause, when his own use comes
first (``wait_for_free_gpu`` brings back the old rule, a GPU job waiting for low utilisation and memory).

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

The machine's owner controls the agent through ``.agent/control.json`` (written by the Windows tray icon,
``pod/agent/windows/tray.ps1``): ``paused`` freezes the running job (SIGSTOP to its process group: the GPU
is free at once, its memory stays allocated) and takes no new job until resumed; ``quota_gb`` caps the
work folder, his cache. Over the cap a running job is stopped and only ``clean`` jobs start: what to delete
is decided on our side, from the per-folder sizes in the heartbeat. For the tray, the agent keeps
``.agent/status.json`` (its state, every tick) and ``.agent/stats.json`` (seconds of job time per hour,
paused time excluded).
"""

from __future__ import annotations

import json
import os
import shutil
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


def _signal_group(proc, sig) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except OSError:
        pass


def _kill(proc) -> None:
    """SIGTERM to the job's whole process group, SIGKILL 30 s later if it is still there."""
    if hasattr(os, "killpg"):
        try:
            pg = os.getpgid(proc.pid)
        except OSError:
            return
        _signal_group(proc, signal.SIGCONT)  # a frozen job only acts on SIGTERM once it runs again
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


def _log_grew(logf: Path, known: int) -> tuple[int, bool]:
    """(length, grew) of a log the child appends to through its own file descriptor. The length comes from
    reading past the known end, never from ``stat``: on the work folder of a WSL2 agent (a Windows drive,
    9P) ``st_mtime``/``st_size`` stay frozen at the first write while the child keeps writing, which made
    the stall guard kill healthy jobs exactly ``stall_minutes`` after they started."""
    try:
        with logf.open("rb") as f:
            f.seek(known)
            new = len(f.read())
    except OSError:
        return known, False
    return known + new, new > 0


class Agent:
    def __init__(self, store, workdir: str | Path, repo: str | Path, name: str | None = None,
                 bin_dir: str | Path | None = None, gpu=gpu_stats, tick: float = 10.0, sync_every: float = 30.0,
                 wait_for_free_gpu: bool = False,
                 free_util: int = 10, free_mem_mib: int = 1500, free_samples: int = 3, free_gap_s: float = 5.0,
                 quota_gb: float = 50.0, disk_every: float = 60.0, log=print):
        self.store, self.workdir, self.repo = store, Path(workdir), Path(repo)
        self.name = name or socket.gethostname()
        self.bin_dir = Path(bin_dir or Path(sys.executable).parent)
        self.gpu, self.tick, self.sync_every, self.log = gpu, tick, sync_every, log
        self.wait_for_free_gpu = wait_for_free_gpu
        self.free = (free_util, free_mem_mib, free_samples, free_gap_s)
        self.state_dir = self.workdir / ".agent"
        (self.state_dir / "finished").mkdir(parents=True, exist_ok=True)
        (self.workdir / "logs").mkdir(exist_ok=True)
        self.online = True
        self.restart = False
        self.quota_gb, self.disk_every = quota_gb, disk_every
        self._disk: tuple[float, int, dict[str, int]] = (-1e18, 0, {})  # (measured at, bytes, top entries)
        self._work: dict[str, float] = {}  # job seconds per epoch hour, not yet added to stats.json
        self._job_start: float | None = None

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
        used, top = self.disk()
        self._local(state, job)
        self._try(self.store.put, f"agents/{self.name}.json", json.dumps(
            {"time": time.time(), "state": state, "job": job, "gpu": g, "version": self._version(),
             "hardware": self.hardware(),
             "paused": self.control()["paused"],
             "disk": {"used_gb": round(used / 1e9, 2), "quota_gb": self.quota(),
                      "top_gb": {k: round(v / 1e9, 2) for k, v in top.items()}}}).encode())

    def hardware(self) -> dict:
        """What jobs can be sized on (measured once; free disk every time): GPUs, CPUs this process may use
        (affinity, capped by a cgroup quota), RAM as this OS sees it (under WSL: the WSL limit)."""
        if not hasattr(self, "_hw"):
            hw: dict = {"gpus": [], "cpus": os.cpu_count()}
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=30).stdout
                hw["gpus"] = [{"name": n.strip(), "vram_mib": int(m), "driver": d.strip()}
                              for n, m, d in (line.split(",") for line in out.strip().splitlines())]
            except Exception:
                pass
            if hasattr(os, "sched_getaffinity"):
                hw["cpus"] = len(os.sched_getaffinity(0))
            try:  # cgroup v2 quota, "max 100000" when unlimited
                quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
                if quota != "max":
                    hw["cpus"] = min(hw["cpus"], max(1, int(int(quota) / int(period))))
            except (OSError, ValueError):
                pass
            try:
                kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemTotal:"))
                hw["ram_gb"] = round(kb / 1e6, 1)
            except (OSError, StopIteration, ValueError):
                pass
            try:  # from the package metadata: importing torch would cost the agent hundreds of MB for good
                from importlib.metadata import version

                hw["torch"] = version("torch")
            except Exception:
                pass
            self._hw = hw
        try:
            free = shutil.disk_usage(self.workdir).free
        except OSError:
            free = None
        return {**self._hw, "disk_free_gb": round(free / 1e9, 1) if free is not None else None}

    def _version(self) -> str:
        try:
            return subprocess.run(["git", "-C", str(self.repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "?"

    # ------------------------------------------------------------------------ owner control, local state

    def control(self) -> dict:
        """The owner's settings from the tray; a missing or half-written file means the defaults."""
        try:  # utf-8-sig: Windows tools may write a BOM
            c = json.loads((self.state_dir / "control.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            c = {}
        return {"paused": bool(c.get("paused", False)), "quota_gb": float(c.get("quota_gb") or self.quota_gb)}

    def quota(self) -> float:
        return self.control()["quota_gb"]

    def disk(self, force: bool = False) -> tuple[int, dict[str, int]]:
        """Bytes in the work folder and its 15 largest top-level entries, measured at most every
        ``disk_every`` seconds (walking a Windows drive from WSL is slow)."""
        if force or time.time() - self._disk[0] >= self.disk_every:
            sizes = {e.name: _size(e) for e in os.scandir(self.workdir)}
            top = dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:15])
            self._disk = (time.time(), sum(sizes.values()), top)
        return self._disk[1], self._disk[2]

    def cache_full(self, force: bool = False) -> bool:
        return self.disk(force)[0] > self.quota() * 1e9

    def _local(self, state: str, job: str | None = None, job_start: float | None = None) -> None:
        _write_json(self.state_dir / "status.json", {
            "time": time.time(), "state": state, "job": job,
            "job_start": job_start or (self._job_start if job else None), "name": self.name,
            "online": self.online, "used_bytes": self.disk()[0], "quota_gb": self.quota()})

    def _count_work(self, seconds: float, flush: bool = False) -> None:
        """Add job time to the current hour; ``flush`` writes the pending hours to stats.json."""
        if seconds > 0:
            hour = str(int(time.time() // 3600))
            self._work[hour] = self._work.get(hour, 0.0) + seconds
        if flush and self._work:
            path = self.state_dir / "stats.json"
            try:
                st = json.loads(path.read_text())
            except (OSError, ValueError):
                st = {}
            hours = st.setdefault("hours", {})
            for h, s in self._work.items():
                hours[h] = round(hours.get(h, 0.0) + s, 1)
            _write_json(path, st)
            self._work = {}

    def publish_tray(self) -> None:
        """Hand this checkout's tray script to the Windows tray, which replaces itself when it changes: an
        ``update`` job (new code, agent restart) thus updates the tray too."""
        src = self.repo / "pod" / "agent" / "windows" / "tray.ps1"
        if not src.exists():
            return
        data = src.read_bytes()
        dst = self.state_dir / "tray.ps1"
        if not dst.exists() or dst.read_bytes() != data:
            tmp = dst.with_name("tray.ps1.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            self.log("tray script published for the owner's tray")
        self._legacy_tray(data)

    def _legacy_tray(self, data: bytes) -> None:
        """Trays installed before self-update never look at .agent/tray.ps1: overwrite their file in
        %LOCALAPPDATA% through WSL interop (taken at the owner's next logon). Best effort: no interop, no copy."""
        if not Path("/mnt/c").exists():
            return
        try:
            win = subprocess.run(["cmd.exe", "/c", "echo %LOCALAPPDATA%"], capture_output=True, text=True,
                                 timeout=15, cwd="/mnt/c").stdout.strip()
            local = subprocess.run(["wslpath", "-a", win], capture_output=True, text=True, timeout=15).stdout.strip()
            target = Path(local) / "priism-agent" / "tray.ps1"
            if target.exists() and target.read_bytes() != data:
                target.write_bytes(data)
                self.log(f"tray script updated in {win}\\priism-agent (applies at next logon)")
        except Exception as e:
            self.log(f"legacy tray not updated ({e!r})")

    def _sleep(self, seconds: float) -> None:
        """Sleep, but wake as soon as the owner changes something in the tray."""
        path = self.state_dir / "control.json"
        stamp = lambda: path.stat().st_mtime if path.exists() else 0.0
        before, end = stamp(), time.time() + seconds
        while time.time() < end:
            time.sleep(min(2.0, max(0.0, end - time.time())))
            if stamp() != before:
                return

    # ---------------------------------------------------------------------------------------------- loop

    def run_forever(self, idle_poll: float = 60.0) -> None:
        self.log(f"agent {self.name} on {self.workdir}")
        self.publish_tray()
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
                self._sleep(idle_poll)

    def step(self) -> bool:
        """One pass; True if a job ran (look again at once), False if idle or waiting."""
        self._resume()
        self._flush()
        if self.control()["paused"]:
            self.heartbeat("paused")
            return False
        try:
            pending = self.store.list("queue/pending/")
        except (StoreError, OSError):
            self.online = False
            self._local("offline")
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
        if job.kind != "clean" and self.cache_full(force=True):
            self.heartbeat("waiting: cache full (clean jobs only)", job.id)
            return False
        if self.wait_for_free_gpu and job.gpu and job.kind != "clean" and not self.gpu_free():
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
        """Watch a job until it ends or must stop. The owner's pauses count for none of the limits: the
        stall and GPU-idle clocks restart at each resume, the time cap counts active time only."""
        idle_since, last_sync, last, paused_s, frozen, resumed = None, 0.0, time.time(), 0.0, False, start
        log_len, log_grew = 0, start  # the log's real length and when it last grew: see _log_grew
        self._job_start = start
        if hasattr(signal, "SIGCONT"):  # a job found frozen after an agent restart: the loop refreezes it if paused
            _signal_group(proc, signal.SIGCONT)
        while True:
            code = proc.poll()
            if code is not None:
                self._count_work(0.0, flush=True)
                return code, "ok" if code == 0 else f"exit code {code}"
            now = time.time()
            dt, last = min(now - last, 60.0), now  # capped: time the PC spent asleep is not job time
            want = self.control()["paused"] and hasattr(signal, "SIGSTOP")
            if want != frozen:
                _signal_group(proc, signal.SIGSTOP if want else signal.SIGCONT)
                frozen, resumed, idle_since, last_sync = want, now, None, 0.0
                msg = "paused by the machine's owner" if frozen else "resumed"
                self.log(f"{job.id}: {msg}")
                with logf.open("a") as f:
                    f.write(f"### agent {time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")
            if frozen:
                paused_s += dt
            else:
                self._count_work(dt)
            self._local("paused" if frozen else "running", job.id, start)
            stop = None
            log_len, grew = _log_grew(logf, log_len)
            if grew:
                log_grew = now
            if frozen:
                pass
            elif now - max(log_grew, resumed) > job.stall_minutes * 60:
                stop = f"no log output for {job.stall_minutes:g} min"
            elif now - start - paused_s > job.max_minutes * 60:
                stop = f"time cap {job.max_minutes:g} min"
            elif job.kind != "clean" and self.cache_full():
                stop = f"cache full ({self.disk()[0] / 1e9:.1f} GB, limit {self.quota():g} GB)"
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
                self._count_work(0.0, flush=True)
                self._sync(job, logf, start, "paused" if frozen else "running")
                try:
                    if self.store.get(f"cancel/{job.id}") is not None:
                        stop = "cancelled"
                except (StoreError, OSError):
                    self.online = False
            if stop:
                self._count_work(0.0, flush=True)
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
        if state in ("running", "paused"):
            self.heartbeat(state, job.id)
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
                # A folder ships only what this job wrote (shared folders hold older runs); a file named
                # explicitly ships whatever its age, so an earlier job's output (a checkpoint) can be fetched.
                root = self.workdir / folder
                paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
                for p in paths:
                    if p.is_file() and (p == root or p.stat().st_mtime >= rec["start"] - 5):
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


def _size(entry: os.DirEntry) -> int:
    """Bytes under a directory entry, symlinks not followed."""
    try:
        if entry.is_dir(follow_symlinks=False):
            return sum(_size(e) for e in os.scandir(entry.path))
        return entry.stat(follow_symlinks=False).st_size
    except OSError:
        return 0


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)
