import json
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
                        free_samples=1, log=lambda *_: None)


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


def test_gpu_busy_waits(tmp_path):
    store, agent = make_agent(tmp_path, gpu=lambda: (95, 9000))
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
