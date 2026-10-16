"""Agent priorities and preemption: a background job gives way to normal jobs, then resumes."""

import json
import signal
import sys
import threading
import time

import pytest
from test_agent import make_agent

from priism.lab.agent import jobs
from priism.lab.agent.agent import _kill
from priism.lab.agent.jobs import Job, JobRejected, validate

PY = sys.executable

# A resumable fake training: counts to 30 from its checkpoint file, one count per 50 ms.
RESUMABLE = ("import pathlib, time\np = pathlib.Path('ckpt.txt')\ni = int(p.read_text()) if p.exists() else 0\n"
             "while i < 30:\n    i += 1; p.write_text(str(i)); print('step', i, flush=True); time.sleep(0.05)")


def fake_commands(monkeypatch, code=RESUMABLE):
    """Background jobs run ``code``; normal ones (the urgent tests) a short print of their own."""
    monkeypatch.setattr("priism.lab.agent.agent.commands", lambda j, *_: [
        [PY, "-c", code if j.priority == "background" or j.id == "first" else "print('urgent')"]])


def test_priority_is_validated_and_defaults_to_normal():
    validate(Job("phase1", ["probe"], priority="background"))
    with pytest.raises(JobRejected):
        validate(Job("phase1", ["probe"], priority="urgent"))
    legacy = Job.from_json(json.dumps({"kind": "phase1", "args": ["probe"], "id": "x", "future_field": 1}).encode())
    assert legacy.priority == "normal" and legacy.rank > Job("phase1", priority="background").rank


def test_scheduler_takes_highest_priority_then_oldest(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path)
    fake_commands(monkeypatch, "print('ok')")
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="a-bg", priority="background"))
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="b-normal"))
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="c-normal"))
    store.put("queue/pending/d-legacy.json", json.dumps({"kind": "phase1", "args": ["probe"], "id": "d-legacy",
                                                          "gpu": False}).encode())  # from an older client
    order = []
    while agent.step():
        order.append(next(k for k in store.list("queue/done/") if k not in order))
    assert [k.split("/")[-1][:-5] for k in order] == ["b-normal", "c-normal", "d-legacy", "a-bg"]


def test_background_job_is_preempted_then_resumes(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path)
    fake_commands(monkeypatch)
    bg = Job("phase1", ["probe"], gpu=False, id="bg", priority="background", results=["ckpt.txt"])
    jobs.submit(store, bg)
    threading.Timer(0.6, lambda: jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="urgent"))).start()
    assert agent.step()  # starts bg, preempted by "urgent" once it is submitted

    spec = Job.from_json(store.get("queue/pending/bg.json"))
    assert spec.preemptions == 1 and 0 < spec.active_minutes < 0.05
    assert store.get("queue/running/bg.json") is None
    st = json.loads(store.get("status/bg.json"))
    assert st["state"] == "preempted" and st["reason"] == "preempted by urgent" and st["synced"]
    saved = int(store.get("results/bg/ckpt.txt"))  # progress synced at the preemption
    assert 0 < saved < 30
    shown = {j["id"]: j for j in jobs.status(store)["jobs"]}
    assert shown["bg"]["state"] == "preempted" and shown["bg"]["priority"] == "background"
    assert shown["bg"]["preemptions"] == 1 and shown["urgent"]["state"] == "pending"

    assert agent.step()  # the normal job runs first...
    assert store.get("queue/done/urgent.json") is not None and store.get("queue/pending/bg.json") is not None
    assert agent.step()  # ...then the background job, from its checkpoint
    assert json.loads(store.get("status/bg.json"))["state"] == "done"
    assert (agent.workdir / "ckpt.txt").read_text() == "30"
    log = jobs.log(store, "bg")
    lines = [line.strip() for line in log.splitlines()]
    assert "preempted by urgent" in log and lines.count("step 1") == 1 and "step 30" in lines  # appended, resumed
    assert not agent.step()


def test_normal_job_is_never_preempted(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path)
    fake_commands(monkeypatch)
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="first"))
    threading.Timer(0.3, lambda: jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="second"))).start()
    assert agent.step()
    assert json.loads(store.get("status/first.json"))["state"] == "done"


def test_rejected_pending_job_does_not_preempt(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path)
    fake_commands(monkeypatch)
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="bg", priority="background"))
    threading.Timer(0.3, lambda: store.put("queue/pending/bad.json", json.dumps(
        {"kind": "bash", "args": ["-c", "id"], "id": "bad"}).encode())).start()
    assert agent.step()
    assert json.loads(store.get("status/bg.json"))["state"] == "done"


def test_time_cap_counts_earlier_runs(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path)
    fake_commands(monkeypatch)
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="bg", priority="background", max_minutes=1.0,
                           active_minutes=0.995))  # 0.3 s left of its budget
    assert agent.step()
    st = json.loads(store.get("status/bg.json"))
    assert st["state"] == "failed" and st["reason"].startswith("time cap")


def test_cancel_a_preempted_job(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path)
    fake_commands(monkeypatch)
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="bg", priority="background"))
    threading.Timer(0.4, lambda: jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="urgent"))).start()
    agent.step()
    jobs.cancel(store, "bg")
    assert store.get("queue/pending/bg.json") is None and store.get("queue/failed/bg.json") is not None
    assert json.loads(store.get("status/bg.json"))["reason"] == "cancelled while preempted"
    assert agent.step() and not agent.step()  # only "urgent" runs

    # cancelled while the agent still held the preempted outcome (offline): it ends failed, not re-queued
    store2, agent2 = make_agent(tmp_path / "b")
    job = Job("phase1", ["probe"], gpu=False, id="bg2", priority="background")
    jobs.submit(store2, job)
    store2.put("queue/running/bg2.json", job.to_json())
    store2.delete("queue/pending/bg2.json")
    jobs.cancel(store2, "bg2")  # not pending: a marker
    agent2._finish(job, -1, "preempted by x", time.time())
    assert store2.get("queue/pending/bg2.json") is None and store2.get("queue/failed/bg2.json") is not None
    assert json.loads(store2.get("status/bg2.json"))["reason"] == "cancelled while preempted"


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="process groups are POSIX only")
def test_kill_lets_the_leader_save_then_sweeps_its_group(tmp_path):
    import subprocess

    code = ("import signal, subprocess, sys, time, pathlib\n"
            "child = subprocess.Popen(['sleep', '60'])\n"
            "def term(*_):\n    pathlib.Path('saved').write_text(str(child.pid)); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, term)\nprint('ready', flush=True)\ntime.sleep(60)")
    proc = subprocess.Popen([PY, "-c", code], cwd=tmp_path, stdout=subprocess.PIPE, start_new_session=True)
    assert proc.stdout.readline().strip() == b"ready"
    t = time.time()
    _kill(proc, grace=10)
    assert time.time() - t < 10 and proc.poll() == 0  # the leader saved and exited on its own
    child = int((tmp_path / "saved").read_text())
    time.sleep(0.2)
    with pytest.raises(OSError):  # the leftover child of the group is gone too
        import os

        os.kill(child, 0)
