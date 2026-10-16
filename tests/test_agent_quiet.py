"""Agent quiet mode: the running job is duty-cycled (frozen most of the time) during quiet hours."""

import calendar
import json
import signal
import time

import pytest
from test_agent import PY, control, make_agent, run

from priism.lab.agent import jobs
from priism.lab.agent.jobs import Job


def at(hhmm: str) -> float:
    """An epoch at that UTC time of day (with utc_offset_min = 0, the PC's local time)."""
    h, m = map(int, hhmm.split(":"))
    return calendar.timegm((2026, 10, 16, h, m, 0))


def test_quiet_hours_and_modes(tmp_path):
    _, agent = make_agent(tmp_path)
    agent.quiet_default = "auto"  # the real agent's default (make_agent turns quiet hours off)
    ctl = agent.control()  # no control.json: auto, 22:00 -> 08:00, 2 s / 3 s
    assert (ctl["quiet"], ctl["quiet_from"], ctl["quiet_to"], ctl["quiet_run_s"], ctl["quiet_pause_s"]) == \
        ("auto", "22:00", "08:00", 2.0, 3.0)
    control(agent, utc_offset_min=0)
    for hhmm, quiet in [("21:59", False), ("22:00", True), ("03:00", True), ("07:59", True), ("08:00", False),
                        ("12:00", False)]:
        assert agent.quiet_now(now=at(hhmm)) is quiet, hhmm
    control(agent, utc_offset_min=120)  # the PC is UTC+2: 21:00 UTC is 23:00 there
    assert agent.quiet_now(now=at("21:00")) and not agent.quiet_now(now=at("06:00"))
    control(agent, utc_offset_min=0, quiet_from="13:00", quiet_to="14:30")  # a window within the day
    assert agent.quiet_now(now=at("14:00")) and not agent.quiet_now(now=at("23:00"))
    control(agent, utc_offset_min=0, quiet_from="22:00", quiet_to="22:00")  # equal bounds: never
    assert not agent.quiet_now(now=at("22:00"))
    control(agent, utc_offset_min=0, quiet_from="late", quiet_to="08:00")  # unreadable: never
    assert not agent.quiet_now(now=at("23:00"))
    control(agent, quiet="on", quiet_from="13:00", quiet_to="14:00", utc_offset_min=0)
    assert agent.quiet_now(now=at("12:00"))
    control(agent, quiet="off")
    assert not agent.quiet_now(now=at("23:00"))
    control(agent, quiet="loud", quiet_run_s="x", quiet_pause_s=0)  # bad values: the default, or clamped
    ctl = agent.control()
    assert ctl["quiet"] == "auto" and ctl["quiet_run_s"] == 2.0 and ctl["quiet_pause_s"] == 0.1


def test_quiet_is_reported_to_tray_and_laptop(tmp_path):
    store, agent = make_agent(tmp_path)
    control(agent, quiet="on")
    agent.heartbeat("idle")
    hb = json.loads(store.get("agents/pc.json"))
    assert hb["quiet"] and hb["quiet_mode"] == "on"
    assert json.loads((agent.state_dir / "status.json").read_text())["quiet"]
    control(agent, quiet="off")
    agent.heartbeat("idle")
    assert not json.loads(store.get("agents/pc.json"))["quiet"]


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="SIGSTOP is POSIX only")
def test_quiet_duty_cycles_the_job_and_limits_ignore_frozen_time(tmp_path):
    store, agent = make_agent(tmp_path)
    control(agent, quiet="on", quiet_run_s=0.2, quiet_pause_s=0.3)  # 40 % duty
    # ~1 s of work, ~2.5 s of wall time: a 1.5 s cap and a 0.3 s stall limit (the frozen parts) would break on a wall clock
    job = Job("phase1", ["probe"], gpu=False, id="q", max_minutes=0.025, stall_minutes=0.005)
    code = "import time\nfor i in range(20):\n    print(i, flush=True); time.sleep(0.05)"
    t = time.time()
    st = run(agent, store, job, code)
    wall = time.time() - t
    assert st["state"] == "done", st
    assert wall > 1.8  # frozen 60 % of the time: ~2.5 s for ~1 s of work
    worked = sum(json.loads((agent.state_dir / "stats.json").read_text())["hours"].values())
    assert worked < 0.7 * wall  # only the running parts count as work
    log = jobs.log(store, "q")
    assert "quiet mode on (runs 0.2 s, frozen 0.3 s)" in log


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="SIGSTOP is POSIX only")
def test_owner_pause_wins_over_quiet(tmp_path):
    import threading

    store, agent = make_agent(tmp_path)
    control(agent, quiet="on", quiet_run_s=0.2, quiet_pause_s=0.3, paused=True)
    threading.Timer(1.5, lambda: control(agent, quiet="on", quiet_run_s=0.2, quiet_pause_s=0.3)).start()
    job = Job("phase1", ["probe"], gpu=False, id="pq")
    code = "import time\nfor i in range(10):\n    print(i, flush=True); time.sleep(0.05)"
    st = run(agent, store, job, code)
    assert st["state"] == "done", st
    log = jobs.log(store, "pq")
    assert log.index("paused by the machine's owner") < log.index("resumed") < log.index("quiet mode on")
    assert PY  # the job ran with the test's interpreter


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="SIGSTOP is POSIX only")
def test_quiet_job_can_still_be_preempted(tmp_path, monkeypatch):
    import threading

    from test_agent_priority import fake_commands

    store, agent = make_agent(tmp_path)
    fake_commands(monkeypatch)
    control(agent, quiet="on", quiet_run_s=0.1, quiet_pause_s=0.15)
    jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="bg", priority="background"))
    threading.Timer(0.5, lambda: jobs.submit(store, Job("phase1", ["probe"], gpu=False, id="urgent"))).start()
    assert agent.step()
    assert json.loads(store.get("status/bg.json"))["state"] == "preempted"
    assert agent.step() and agent.step()  # urgent, then bg resumed and finished, still quiet
    assert json.loads(store.get("status/bg.json"))["state"] == "done"
    assert (agent.workdir / "ckpt.txt").read_text() == "30"
