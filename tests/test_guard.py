import importlib.util
import os
import subprocess
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("guard", Path(__file__).parents[1] / "pod" / "guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def _queue(tmp_path, name, text, log_age_s):
    for d in ("running", "logs"):
        (tmp_path / d).mkdir(exist_ok=True)
    s = tmp_path / "running" / name
    s.write_text(text)
    log = tmp_path / "logs" / name.replace(".sh", ".log")
    log.write_text("x\n")
    t = time.time() - log_age_s
    os.utime(s, (t, t))
    os.utime(log, (t, t))


def test_guard_stops_silent_and_gpu_idle_jobs(tmp_path):
    stopped = []
    stopper = lambda q, s, r: stopped.append((s.name, r))
    _queue(tmp_path, "0001-silent.sh", "echo\n", 25 * 60)  # gpu lane, limit 20 min
    _queue(tmp_path, "0002-data.sh", "# lane: cpu\n", 25 * 60)  # cpu lane, limit 60 min
    _queue(tmp_path, "0003-slow.sh", "# lane: cpu\n# stall: 10\n", 11 * 60)
    assert guard.check(tmp_path, {}, 15, util=lambda: 50, stopper=stopper) == ["0001-silent.sh", "0003-slow.sh"]
    for f in (tmp_path / "running").iterdir():
        f.unlink()
    _queue(tmp_path, "0004-train.sh", "# stall: 999\n", 60)
    idle, now = {}, time.time()
    run = lambda dt, u: guard.check(tmp_path, idle, 15, now=now + dt * 60, util=lambda: u, stopper=stopper)
    assert run(0, 0) == [] and run(10, 0) == []
    assert run(11, 30) == []  # busy again: the idle clock restarts
    assert run(12, 0) == [] and run(26, 0) == []
    assert run(28, 0) == ["0004-train.sh"] and stopped[-1][1] == "GPU idle for 15 min"

def test_guard_kills_the_job_process_group(tmp_path, monkeypatch):
    _queue(tmp_path, "0005-hang.sh", "sleep 300 & wait\n", 0)
    script = tmp_path / "running" / "0005-hang.sh"
    proc = subprocess.Popen(["bash", str(script)], start_new_session=True)
    for _ in range(100):  # until the job's bash shows up in /proc with its script argument
        if guard.job_pgids(script):
            break
        time.sleep(0.05)
    monkeypatch.setattr(guard.time, "sleep", lambda s: None)
    guard.stop(tmp_path, script, "test", log=lambda m: None)
    assert proc.wait(timeout=10) != 0
    assert "guard" in (tmp_path / "logs" / "0005-hang.log").read_text()
