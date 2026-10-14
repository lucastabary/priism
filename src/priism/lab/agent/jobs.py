"""Job specs, the allowlist, and the client side of the queue.

A job is declarative JSON, never a script: a ``kind`` from the allowlist plus string arguments. The agent
builds the command itself and runs it without a shell, inside its work folder. Whoever can write to the
queue can therefore only run priism commands with relative paths, not arbitrary code on the machine
(the ``update`` kind does run code: the priism repository's own, at a branch of its ``origin``).
``clean`` deletes paths inside the work folder, the owner's size-capped cache: the agent reports its
per-folder sizes in its heartbeat, and we decide here what goes.

Store layout: ``queue/{pending,running,done,failed}/<id>.json`` (the spec, then its outcome),
``logs/<id>.log``, ``status/<id>.json`` (heartbeat, state, last line), ``cancel/<id>`` (a marker the
agent obeys), ``results/<id>/...`` (the files the job wrote under its ``results`` folders),
``inputs/<folder>/...`` (data a job lists in ``inputs``: downloaded into ``<folder>`` of the work folder
before it runs, files already there skipped; put them there with any S3 client),
``agents/<name>.json`` (each agent's heartbeat).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

# priism-lab subcommands a remote job may run.
LAB_COMMANDS = {"gen", "cache-features", "registry-eval", "train-id", "eval-sep", "eval-stems",
                "registry-encoders", "registry-domain", "registry-dims", "registry-separate-eval"}
SCRIPTS = {"phase1": ("scripts/registry_phase1.sh", {"probe", "full"})}
KINDS = {"priism-lab", "update", "clean", *SCRIPTS}
_ENV_KEY = re.compile(r"^PRIISM_[A-Z0-9_]+$")
_ENV_VAL = re.compile(r"^[\w.:,=-]{0,200}$")
PROTECTED = {".agent"}  # the agent's own state (control, stats, unsynced outcomes): never cleaned
_BRANCH = re.compile(r"^[\w][\w./-]{0,100}$")


class JobRejected(ValueError):
    pass


@dataclass
class Job:
    kind: str
    args: list[str] = field(default_factory=list)
    id: str = ""
    gpu: bool = True
    max_minutes: float = 240.0
    stall_minutes: float = 20.0  # no new log line for this long: stopped (lab rule)
    gpu_idle_minutes: float = 15.0  # a GPU job whose GPU sits at 0 % this long: stopped
    results: list[str] = field(default_factory=list)  # folders (what the job wrote) or files (relative), uploaded at the end
    env: dict[str, str] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)  # folders fetched from the store's inputs/ before the job
    created: float = 0.0

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), indent=1).encode()

    @classmethod
    def from_json(cls, data: bytes) -> Job:
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _safe_rel(arg: str) -> None:
    """Paths in arguments stay inside the work folder: no absolute path, no '..', no home or drive."""
    for parts in (PurePosixPath(arg).parts, PureWindowsPath(arg).parts):
        if ".." in parts:
            raise JobRejected(f"argument escapes the work folder: {arg!r}")
    if arg.startswith(("/", "~", "\\")) or re.match(r"^[A-Za-z]:", arg) or "\0" in arg:
        raise JobRejected(f"absolute or home path not allowed: {arg!r}")


def validate(job: Job) -> None:
    if job.kind not in KINDS:
        raise JobRejected(f"unknown kind {job.kind!r} (allowed: {sorted(KINDS)})")
    if not all(isinstance(a, str) for a in job.args):
        raise JobRejected("arguments must be strings")
    for a in job.args + job.results + job.inputs:
        _safe_rel(a)
    for k, v in job.env.items():
        if not _ENV_KEY.match(k) or not _ENV_VAL.match(str(v)):
            raise JobRejected(f"environment variable not allowed: {k}={v!r}")
    if job.kind == "priism-lab" and (not job.args or job.args[0] not in LAB_COMMANDS):
        raise JobRejected(f"priism-lab command not allowed: {job.args[:1]} (allowed: {sorted(LAB_COMMANDS)})")
    if job.kind in SCRIPTS and job.args not in [[s] for s in SCRIPTS[job.kind][1]]:
        raise JobRejected(f"{job.kind} takes one of {sorted(SCRIPTS[job.kind][1])}")
    if job.kind == "update" and (len(job.args) != 1 or not _BRANCH.match(job.args[0]) or ".." in job.args[0]):
        raise JobRejected("update takes one branch name")
    if job.kind == "clean":
        for a in job.args:
            parts = [p for p in PurePosixPath(a.replace("\\", "/")).parts if p != "."]
            if not parts or parts[0] in PROTECTED:
                raise JobRejected(f"clean cannot delete {a!r} (the work folder itself or {sorted(PROTECTED)})")
        if not job.args:
            raise JobRejected("clean takes the paths to delete")


def commands(job: Job, repo: Path, bin_dir: Path) -> list[list[str]]:
    """The command lines a valid job runs, in order (no shell)."""
    validate(job)
    exe = lambda name: str(next((p for p in (bin_dir / name, bin_dir / f"{name}.exe") if p.exists()),
                                bin_dir / name))
    if job.kind == "priism-lab":
        return [[exe("priism-lab"), *job.args]]
    if job.kind in SCRIPTS:
        return [["bash", str(repo / SCRIPTS[job.kind][0]), *job.args]]
    if job.kind == "clean":
        return [[exe("python"), "-m", "priism.lab.agent.clean", *job.args]]
    branch = job.args[0]
    return [["git", "-C", str(repo), "fetch", "origin", branch],
            ["git", "-C", str(repo), "checkout", "--detach", f"origin/{branch}"],
            [exe("python"), "-m", "pip", "install", "-q", "-e", f"{repo}[lab,fast]"],  # fast: numba (generator)
            [exe("python"), "-c", "import numba, torch; print('numba', numba.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"]]


# ------------------------------------------------------------------------------------------- client side

def submit(store, job: Job) -> str:
    validate(job)
    job.created = time.time()
    job.id = job.id or time.strftime("%Y%m%d-%H%M%S") + f"-{job.kind}-{uuid.uuid4().hex[:4]}"
    store.put(f"queue/pending/{job.id}.json", job.to_json())
    return job.id


def status(store) -> dict:
    out: dict = {"agents": {}, "jobs": []}
    for key in store.list("agents/"):
        d = json.loads(store.get(key) or b"{}")
        d["age_s"] = time.time() - d.get("time", 0)
        out["agents"][PurePosixPath(key).stem] = d
    for state in ("pending", "running", "done", "failed"):
        for key in store.list(f"queue/{state}/"):
            jid = PurePosixPath(key).stem
            st = json.loads(store.get(f"status/{jid}.json") or b"{}") if state != "pending" else {}
            out["jobs"].append({"id": jid, "state": state, **{k: st.get(k) for k in
                                ("reason", "exit_code", "minutes", "last_line", "time", "synced")}})
    return out


def log(store, job_id: str, tail: int | None = None) -> str:
    text = (store.get(f"logs/{job_id}.log") or b"").decode(errors="replace")
    return "\n".join(text.splitlines()[-tail:]) if tail else text


def cancel(store, job_id: str) -> None:
    if store.get(f"queue/pending/{job_id}.json") is not None:
        store.delete(f"queue/pending/{job_id}.json")
    else:
        store.put(f"cancel/{job_id}", b"cancel")


def pull(store, job_id: str, dest: str | Path) -> list[Path]:
    got = []
    for key in store.list(f"results/{job_id}/"):
        path = Path(dest) / key[len(f"results/{job_id}/"):]
        store.download_file(key, path)
        got.append(path)
    return got
