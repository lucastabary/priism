"""GPU thermal sampling, quiet mode's temperature controller, the gpu-load measurement and its report."""

import json
import signal
import time

import pytest
from test_agent import control, make_agent, run

from priism.lab.agent import jobs, thermal
from priism.lab.agent.jobs import Job, JobRejected, validate
from priism.lab.agent.thermal import next_pause


def test_controller_steps_with_hysteresis():
    args = dict(target=60.0, band=3.0, lo=1.0, hi=20.0)
    assert next_pause(3.0, 65.0, **args) == 4.5  # too hot: longer pauses
    assert next_pause(3.0, 58.0, **args) == 3.0  # within the band: hold
    assert next_pause(3.0, 50.0, **args) == 2.4  # cool: shorter pauses
    assert next_pause(18.0, 90.0, **args) == 20.0 and next_pause(1.1, 30.0, **args) == 1.0  # capped
    p = 3.0
    for _ in range(20):  # stays hot: climbs to the cap, never past it
        p = next_pause(p, 70.0, **args)
    assert p == 20.0


def test_gpu_thermal_parses_and_skips_na(monkeypatch):
    class R:
        def __init__(self, out):
            self.stdout = out

    monkeypatch.setattr(thermal.subprocess, "run", lambda *a, **k: R("57, [N/A], 99\n"))
    assert thermal.gpu_thermal() == {"temp_c": 57.0, "power_w": None, "util": 99.0}
    monkeypatch.setattr(thermal.subprocess, "run", lambda *a, **k: R("[N/A], [N/A], [N/A]\n"))
    assert thermal.gpu_thermal() is None

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(thermal.subprocess, "run", missing)
    assert thermal.gpu_thermal() is None


def test_samples_go_to_files_status_heartbeat_and_results(tmp_path):
    store, agent = make_agent(tmp_path)
    agent.thermal, agent.thermal_every = (lambda: {"temp_c": 55.0, "power_w": 40.0, "util": 97.0}), 0.05
    job = Job("phase1", ["probe"], gpu=False, id="t1")
    st = run(agent, store, job, "import time\nfor i in range(8):\n    print(i, flush=True); time.sleep(0.05)")
    assert st["state"] == "done"
    local = thermal.read_jsonl(agent.state_dir / "thermal.jsonl")
    assert len(local) >= 3 and local[0]["temp_c"] == 55.0 and local[0]["job"] == "t1" and not local[0]["quiet"]
    shipped = [json.loads(line) for line in store.get("results/t1/thermal.jsonl").decode().splitlines()]
    assert len(shipped) >= 3  # the job's own curve, pulled with its results
    assert any(p.name == "thermal.jsonl" for p in jobs.pull(store, "t1", tmp_path / "pulled"))
    summary = agent.thermal_summary()
    assert summary["mean_10min"] == {"temp_c": 55.0, "power_w": 40.0, "util": 97.0}
    assert json.loads((agent.state_dir / "status.json").read_text())["thermal"]["last"]["util"] == 97.0
    agent.heartbeat("idle")
    assert json.loads(store.get("agents/pc.json"))["thermal"]["last"]["temp_c"] == 55.0


def test_quiet_pause_follows_temperature_or_falls_back(tmp_path):
    _, agent = make_agent(tmp_path)
    control(agent, quiet="on")
    ctl = agent.control()
    assert (ctl["quiet_temp_c"], ctl["quiet_pause_min_s"], ctl["quiet_pause_max_s"]) == (60.0, 1.0, 20.0)
    assert agent._quiet_pause(ctl) == (3.0, "ratio")  # no reading yet: the fixed ratio
    now = time.time()
    agent._samples.append({"time": now, "temp_c": 70.0, "power_w": None, "util": 100.0})
    assert agent._quiet_pause(ctl) == (4.5, "temp")
    assert agent._quiet_pause(ctl) == (4.5, "temp")  # same reading: no second step
    agent._samples.append({"time": now + 1, "temp_c": 70.0, "power_w": None, "util": 100.0})
    assert agent._quiet_pause(ctl) == (6.75, "temp")
    agent._samples.append({"time": now - 3600, "temp_c": 70.0, "power_w": None, "util": 100.0})  # stale
    assert agent._quiet_pause(ctl) == (3.0, "ratio")
    agent._samples.append({"time": time.time(), "temp_c": None, "power_w": 30.0, "util": 100.0})  # temp N/A
    assert agent._quiet_pause(ctl) == (3.0, "ratio")
    control(agent, quiet="on", quiet_temp_c=0)  # target off: always the fixed ratio
    agent._samples.append({"time": time.time() + 2, "temp_c": 90.0, "power_w": None, "util": 100.0})
    assert agent._quiet_pause(agent.control()) == (3.0, "ratio")


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="SIGSTOP is POSIX only")
@pytest.mark.parametrize("temp", [75.0, 40.0])
def test_quiet_job_pauses_drift_to_the_temperature(tmp_path, temp):
    store, agent = make_agent(tmp_path)
    agent.thermal, agent.thermal_every = (lambda: {"temp_c": temp, "power_w": 30.0, "util": 90.0}), 0.05
    control(agent, quiet="on", quiet_run_s=0.1, quiet_pause_s=0.2, quiet_pause_min_s=0.1, quiet_pause_max_s=0.4)
    job = Job("phase1", ["probe"], gpu=False, id="hot")
    st = run(agent, store, job, "import time\nfor i in range(30):\n    print(i, flush=True); time.sleep(0.05)")
    assert st["state"] == "done", st
    quiet = [s for s in thermal.read_jsonl(agent.workdir / "logs" / "hot.thermal.jsonl") if s["quiet"]]
    assert quiet and quiet[-1]["mode"] == "temp"
    if temp > 60:  # too hot: up to the longest pause
        assert quiet[-1]["pause_s"] == pytest.approx(0.4)
    else:  # cool: shorter than where it started
        assert quiet[-1]["pause_s"] < 0.2
    assert "to keep the GPU under 60 C" in jobs.log(store, "hot")


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="SIGSTOP is POSIX only")
def test_job_quiet_setting_applies_only_while_the_owner_says_auto(tmp_path):
    code = "import time\nfor i in range(6):\n    print(i, flush=True); time.sleep(0.05)"
    store, agent = make_agent(tmp_path)
    control(agent, quiet="auto", quiet_from="00:00", quiet_to="00:00", quiet_temp_c=0, quiet_run_s=0.1,
            quiet_pause_s=0.1)  # auto, but no quiet hours
    run(agent, store, Job("phase1", ["probe"], gpu=False, id="j-on", quiet="on"), code)
    assert "quiet mode on" in jobs.log(store, "j-on")
    control(agent, quiet="off", quiet_temp_c=0)  # the owner's never-quiet wins over the job
    run(agent, store, Job("phase1", ["probe"], gpu=False, id="j-on2", quiet="on"), code)
    assert "quiet mode on" not in jobs.log(store, "j-on2")
    with pytest.raises(JobRejected):
        validate(Job("phase1", ["probe"], quiet="loud"))


def test_quiet_setting_is_validated():
    validate(Job("priism-lab", ["gpu-load", "--minutes", "10"], quiet="off"))
    with pytest.raises(JobRejected):
        validate(Job("phase1", ["probe"], quiet="loud"))


def test_gpu_load_on_cpu_and_report(tmp_path, capsys):
    out = thermal.gpu_load(minutes=0.01, size=64, device="cpu", report_every=0.2)
    assert out["products"] > 0 and out["mean_tflops"] >= 0
    assert '"tflops"' in capsys.readouterr().out
    t0 = 1_000_000.0
    normal = [{"time": t0 + 10 * i, "temp_c": 60.0 + i, "power_w": 80.0, "util": 100.0, "quiet": False,
               "pause_s": 0.0} for i in range(61)]  # 10 minutes
    quiet = [{"time": t0 + 10 * i, "temp_c": 50.0, "power_w": None, "util": 40.0, "quiet": True, "pause_s": 3.0}
             for i in range(61)]
    for name, rows in (("normal.jsonl", normal), ("quiet.jsonl", quiet)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"cut')
    a, b = thermal.report([str(tmp_path / "normal.jsonl"), str(tmp_path / "quiet.jsonl")], last_minutes=5)
    assert a["minutes"] == 10.0 and a["max_temp_c"] == 120.0 and a["quiet_share"] == 0.0
    assert a["last5min_temp_c"] == 105.0  # samples 30..60 (90..120 C): the steady state
    assert b["mean_power_w"] is None and b["quiet_share"] == 1.0 and b["mean_pause_s"] == 3.0
