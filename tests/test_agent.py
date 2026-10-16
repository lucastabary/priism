import json
import os
from pathlib import Path
import sys

import pytest

from priism.lab.agent import jobs
from priism.lab.agent.agent import Agent
from priism.lab.agent.jobs import Job, JobRejected, validate
from priism.lab.agent.store import LocalStore, StoreError

PY = sys.executable


def make_agent(tmp_path, store=None, gpu=lambda: (0, 0)):
    store = store or LocalStore(tmp_path / "store")
    return store, Agent(store, tmp_path / "work", tmp_path, name="pc", gpu=gpu, tick=0.05, sync_every=0.1,
                        free_samples=1, quiet_default="off", log=lambda *_: None)  # no quiet hours unless asked


def test_allowlist():
    validate(Job("priism-lab", ["registry-eval", "--cache", "cache", "--tune", "ni%dev"]))
    validate(Job("phase1", ["probe"]))
    validate(Job("update", ["lab/registry"]))
    for bad in [Job("bash", ["-c", "rm -rf ~"]), Job("priism-lab", ["worker", "--queue", "q"]),
                Job("priism-lab", ["gen", "--out", "/etc"]), Job("priism-lab", ["gen", "--out", "../x"]),
                Job("priism-lab", ["gen", "--out", "C:\\Users"]), Job("phase1", ["full", "--x"]),
                Job("update", ["main; rm"]), Job("priism-lab", ["gen"], env={"LD_PRELOAD": "x"}),
                Job("priism-lab", ["gen"], results=["../../home"])]:
        with pytest.raises(JobRejected):
            validate(bad)


def test_submit_status_cancel_pending(tmp_path):
    store = LocalStore(tmp_path)
    jid = jobs.submit(store, Job("phase1", ["probe"]))
    st = jobs.status(store)
    assert [j["id"] for j in st["jobs"]] == [jid] and st["jobs"][0]["state"] == "pending"
    jobs.cancel(store, jid)
    assert jobs.status(store)["jobs"] == []


def test_rejected_job_goes_to_failed(tmp_path):
    store, agent = make_agent(tmp_path)
    store.put("queue/pending/x.json", json.dumps({"kind": "bash", "args": ["-c", "id"], "id": "x"}).encode())
    assert agent.step()
    assert store.get("queue/failed/x.json") is not None
    assert "rejected" in json.loads(store.get("status/x.json"))["reason"]


def test_gpu_busy_starts_anyway_unless_asked_to_wait(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path, gpu=lambda: (95, 9000))
    jid = jobs.submit(store, Job("phase1", ["probe"], gpu=False))  # gpu=False: no GPU-idle stop in the test
    monkeypatch.setattr("priism.lab.agent.agent.commands", lambda j, *_: [[PY, "-c", "print('ran')"]])
    assert agent.step()  # the owner's GPU use does not hold the job back; only his pause does
    assert store.get(f"queue/done/{jid}.json") is not None

    agent.wait_for_free_gpu = True
    jid = jobs.submit(store, Job("phase1", ["probe"]))
    assert not agent.step()
    assert store.get(f"queue/pending/{jid}.json") is not None  # still waiting
    assert json.loads(store.get("agents/pc.json"))["state"] == "waiting: GPU busy"


def run(agent, store, job, code):
    jobs.submit(store, job)
    store.put(f"queue/running/{job.id}.json", job.to_json())
    store.delete(f"queue/pending/{job.id}.json")
    agent.run_job(job, [[PY, "-c", code]])
    return json.loads(store.get(f"status/{job.id}.json"))


def test_job_runs_logs_and_uploads_results(tmp_path):
    store, agent = make_agent(tmp_path)
    code = "import pathlib; pathlib.Path('results').mkdir(exist_ok=True); pathlib.Path('results/r.txt').write_text('ok'); print('hello')"
    st = run(agent, store, Job("phase1", ["probe"], gpu=False, results=["results"]), code)
    jid = jobs.status(store)["jobs"][0]["id"]
    assert st["state"] == "done" and st["synced"]
    assert "hello" in jobs.log(store, jid)
    assert store.get(f"queue/done/{jid}.json") is not None and store.get(f"queue/running/{jid}.json") is None
    got = jobs.pull(store, jid, tmp_path / "pulled")
    assert [p.read_text() for p in got] == ["ok"]


def test_results_file_ships_even_if_older_than_the_job(tmp_path):
    store, agent = make_agent(tmp_path)
    old = agent.workdir / "runs" / "best.pt"
    old.parent.mkdir(parents=True)
    old.write_text("weights")
    os.utime(old, (1.0, 1.0))  # written long before this job, by an earlier one
    (agent.workdir / "runs" / "stale.txt").write_text("x")
    os.utime(agent.workdir / "runs" / "stale.txt", (1.0, 1.0))
    st = run(agent, store, Job("phase1", ["probe"], gpu=False, id="ship", results=["runs/best.pt"]), "print('ok')")
    assert st["state"] == "done"
    assert store.get("results/ship/runs/best.pt") == b"weights"
    assert store.get("results/ship/runs/stale.txt") is None


def test_stall_and_cancel_stop_the_job(tmp_path):
    store, agent = make_agent(tmp_path)
    st = run(agent, store, Job("phase1", ["probe"], gpu=False, stall_minutes=0.01),
             "import time; print('start', flush=True); time.sleep(30)")
    assert st["state"] == "failed" and "no log output" in st["reason"]

    store2, agent2 = make_agent(tmp_path / "b")
    job = Job("phase1", ["probe"], gpu=False, id="c1")
    store2.put("cancel/c1", b"cancel")
    st = run(agent2, store2, job, "import time\nwhile True:\n    print('tick', flush=True); time.sleep(0.05)")
    assert st["state"] == "failed" and st["reason"] == "cancelled"
    assert store2.get("cancel/c1") is None


class FlakyStore(LocalStore):
    down = False

    def put(self, key, data):
        if self.down:
            raise StoreError("network down")
        super().put(key, data)

    def upload_file(self, key, path):
        if self.down:
            raise StoreError("network down")
        super().upload_file(key, path)


def test_offline_job_finishes_and_syncs_later(tmp_path):
    store = FlakyStore(tmp_path / "store")
    store, agent = make_agent(tmp_path, store=store)
    job = Job("phase1", ["probe"], gpu=False, id="off", results=["results"])
    jobs.submit(store, job)
    store.down = True
    agent.run_job(job, [[PY, "-c", "import pathlib; pathlib.Path('results').mkdir(); pathlib.Path('results/a').write_text('1'); print('done')"]])
    assert not agent.online and store.get("status/off.json") is None
    assert (agent.state_dir / "finished" / "off.json").exists()  # kept until uploaded
    store.down = False
    agent._flush()
    assert json.loads(store.get("status/off.json"))["state"] == "done"
    assert store.get("results/off/results/a") == b"1"
    assert not (agent.state_dir / "finished" / "off.json").exists()


def control(agent, **kw):
    (agent.state_dir / "control.json").write_text("\ufeff" + json.dumps(kw), encoding="utf-8")  # BOM, as Windows


def test_paused_takes_no_job(tmp_path):
    store, agent = make_agent(tmp_path)
    jid = jobs.submit(store, Job("phase1", ["probe"], gpu=False))
    control(agent, paused=True)
    assert not agent.step()
    assert store.get(f"queue/pending/{jid}.json") is not None
    assert json.loads(store.get("agents/pc.json"))["paused"]
    assert json.loads((agent.state_dir / "status.json").read_text())["state"] == "paused"


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGSTOP"), reason="SIGSTOP is POSIX only")
def test_pause_freezes_the_job_and_limits_ignore_it(tmp_path):
    import threading
    import time

    store, agent = make_agent(tmp_path)
    job = Job("phase1", ["probe"], gpu=False, id="p1", stall_minutes=0.01)  # 0.6 s without output = stall
    control(agent, paused=True)
    threading.Timer(2.0, lambda: control(agent, paused=False)).start()  # paused 2 s, longer than the stall limit
    code = "import time\nfor i in range(20):\n    print(i, flush=True); time.sleep(0.05)"
    st = run(agent, store, job, code)
    assert st["state"] == "done", st
    log = jobs.log(store, "p1")
    assert "paused by the machine's owner" in log and "resumed" in log
    worked = sum(json.loads((agent.state_dir / "stats.json").read_text())["hours"].values())
    assert 0 < worked < 1.9  # job time only, the 2 s pause excluded


def test_stats_count_job_time(tmp_path):
    store, agent = make_agent(tmp_path)
    run(agent, store, Job("phase1", ["probe"], gpu=False, id="s1"), "import time; print('x', flush=True); time.sleep(0.5)")
    worked = sum(json.loads((agent.state_dir / "stats.json").read_text())["hours"].values())
    assert 0.3 < worked < 2


def test_cache_full_only_clean_runs(tmp_path, monkeypatch):
    store, agent = make_agent(tmp_path)
    (agent.workdir / "data").mkdir()
    (agent.workdir / "data" / "big.bin").write_bytes(b"x" * 2000)
    control(agent, quota_gb=1e-6)  # 1000 bytes
    jid = jobs.submit(store, Job("phase1", ["probe"], gpu=False))
    assert not agent.step()
    assert store.get(f"queue/pending/{jid}.json") is not None
    hb = json.loads(store.get("agents/pc.json"))
    assert hb["state"].startswith("waiting: cache full") and "data" in hb["disk"]["top_gb"]
    jobs.cancel(store, jid)

    job = Job("clean", ["data/big.bin"], id="c1")
    assert jobs.commands(job, tmp_path, Path(PY).parent)[0][1:3] == ["-m", "priism.lab.agent.clean"]
    jobs.submit(store, job)
    monkeypatch.setattr("priism.lab.agent.agent.commands", lambda j, *_: [[PY, "-m", "priism.lab.agent.clean", *j.args]])
    assert agent.step()  # the cache is full, but a clean job still starts
    assert json.loads(store.get("status/c1.json"))["state"] == "done"
    assert not (agent.workdir / "data" / "big.bin").exists()
    assert not agent.cache_full(force=True)


def test_clean_refuses_outside_and_agent_state(tmp_path):
    for bad in [[], ["."], ["./"], [".agent"], [".agent/control.json"], ["../x"], ["/etc"], ["C:\\x"]]:
        with pytest.raises(JobRejected):
            validate(Job("clean", bad))
    validate(Job("clean", ["cache/synth-train", "data/ni-download"]))


def test_tray_script_published_for_the_owner(tmp_path):
    store, agent = make_agent(tmp_path)
    src = tmp_path / "pod" / "agent" / "windows" / "tray.ps1"  # make_agent: repo = tmp_path
    src.parent.mkdir(parents=True)
    src.write_text("# v1")
    agent.publish_tray()
    assert (agent.state_dir / "tray.ps1").read_text() == "# v1"
    src.write_text("# v2")
    agent.publish_tray()
    assert (agent.state_dir / "tray.ps1").read_text() == "# v2"


def test_heartbeat_reports_hardware(tmp_path):
    store, agent = make_agent(tmp_path)
    agent.heartbeat("idle")
    hw = json.loads(store.get("agents/pc.json"))["hardware"]
    assert hw["cpus"] >= 1 and hw["disk_free_gb"] > 0 and isinstance(hw["gpus"], list)
    validate(Job("priism-lab", ["registry-encoders", "--cache", "cache"]))


def test_inputs_are_fetched_before_the_job_and_kept(tmp_path):
    store, agent = make_agent(tmp_path)
    store.put("inputs/data/set1/a/x.txt", b"x")
    store.put("inputs/data/set1/b.txt", b"b")
    (tmp_path / "work" / "data" / "set1").mkdir(parents=True)
    (tmp_path / "work" / "data" / "set1" / "b.txt").write_text("local")  # already there: kept
    code = "import pathlib; print(pathlib.Path('data/set1/a/x.txt').read_text())"
    st = run(agent, store, Job("phase1", ["probe"], gpu=False, inputs=["data/set1"]), code)
    jid = jobs.status(store)["jobs"][0]["id"]
    assert st["state"] == "done"
    assert "1 files fetched, 1 already there" in jobs.log(store, jid)
    assert (tmp_path / "work" / "data" / "set1" / "b.txt").read_text() == "local"
    with pytest.raises(JobRejected):
        validate(Job("phase1", ["probe"], inputs=["../outside"]))
