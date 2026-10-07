"""Job queue that keeps a rented GPU busy without anyone watching it.

Jobs are shell scripts in ``<queue>/pending``; the worker runs them one at a
time in name order, so the next job starts the moment the previous one
ends. A small HTTP API (bearer token) lets a remote session add jobs, read
logs, check GPU use and download results (checkpoints, logs) through the
provider's HTTPS proxy. Pod disks are not persistent, so results must be
pulled off the pod before it is stopped.

Layout::

    <queue>/pending/0010-make-dataset.sh
    <queue>/running/...   <queue>/done/...   <queue>/failed/...
    <queue>/logs/0010-make-dataset.log
"""

from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

STATES = ("pending", "running", "done", "failed")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}\.sh$")


class Queue:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for d in (*STATES, "logs"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.current: subprocess.Popen | None = None
        self.current_name: str | None = None
        self._lock = threading.Lock()

    def list(self, state: str) -> list[str]:
        return sorted(p.name for p in (self.root / state).glob("*.sh"))

    def add(self, name: str, script: str) -> str:
        if not NAME_RE.match(name):
            raise ValueError("job name must look like 0010-some-job.sh")
        if any((self.root / s / name).exists() for s in STATES):
            raise FileExistsError(name)
        tmp = self.root / "pending" / f".{name}.tmp"
        tmp.write_text(script)
        tmp.rename(self.root / "pending" / name)
        return name

    def cancel(self, name: str) -> str:
        """Drop a pending job, or stop the running one (it ends up in failed)."""
        pending = self.root / "pending" / name
        if pending.exists():
            pending.unlink()
            return "removed"
        with self._lock:
            if self.current_name == name and self.current:
                os.killpg(self.current.pid, signal.SIGTERM)
                return "stopping"
        raise FileNotFoundError(name)

    def log(self, name: str, tail: int = 200) -> str:
        path = self.root / "logs" / name.replace(".sh", ".log")
        if not path.exists():
            raise FileNotFoundError(name)
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200 * tail))
            return b"\n".join(f.read().splitlines()[-tail:]).decode(errors="replace")

    def run_next(self) -> str | None:
        """Run the first pending job to completion; None when the queue is empty."""
        pending = self.list("pending")
        if not pending:
            return None
        name = pending[0]
        running = self.root / "running" / name
        shutil.move(self.root / "pending" / name, running)
        log = (self.root / "logs" / name.replace(".sh", ".log")).open("ab")
        log.write(f"### start {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        log.flush()
        with self._lock:
            self.current = subprocess.Popen(["bash", str(running)], stdout=log, stderr=subprocess.STDOUT,
                                            cwd=self.root, start_new_session=True)
            self.current_name = name
        code = self.current.wait()
        log.write(f"### end {time.strftime('%Y-%m-%d %H:%M:%S')} exit={code}\n".encode())
        log.close()
        with self._lock:
            self.current = self.current_name = None
        shutil.move(running, self.root / ("done" if code == 0 else "failed") / name)
        return name

    def status(self) -> dict:
        return {
            "running": self.list("running"),
            "pending": self.list("pending"),
            "done": self.list("done")[-20:],
            "failed": self.list("failed")[-20:],
            "gpu": gpu_status(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def gpu_status() -> list[dict] | str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"
    keys = ["name", "util_pct", "mem_used_mb", "mem_total_mb", "temp_c"]
    return [dict(zip(keys, (v.strip() for v in line.split(",")))) for line in out.strip().splitlines()]


def resolve_file(root: Path, rel: str) -> Path:
    """Path of ``rel`` inside ``root``; refuses anything that escapes it."""
    root = root.resolve()
    path = (root / rel.lstrip("/")).resolve()
    if path != root and root not in path.parents:
        raise PermissionError(rel)
    return path


def list_files(root: Path, rel: str = "") -> list[dict]:
    base = resolve_file(root, rel)
    if not base.is_dir():
        raise FileNotFoundError(rel)
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            st = p.stat()
            out.append({"path": str(p.relative_to(root.resolve())), "size": st.st_size, "mtime": int(st.st_mtime)})
    return out


def make_handler(queue: Queue, token: str, files_root: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _auth(self) -> bool:
            got = self.headers.get("Authorization", "")
            if hmac.compare_digest(got, f"Bearer {token}"):
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def _send(self, code: int, body: dict | str) -> None:
            data = (json.dumps(body, indent=1) if isinstance(body, dict) else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json" if isinstance(body, dict) else "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            start = 0
            # "Range: bytes=N-" lets a download resume after a dropped connection.
            m = re.match(r"bytes=(\d+)-$", self.headers.get("Range", ""))
            if m and int(m.group(1)) < size:
                start = int(m.group(1))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size - start))
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                shutil.copyfileobj(f, self.wfile, 1 << 20)

        def do_GET(self):  # noqa: N802 - http.server API
            if not self._auth():
                return
            url = urlparse(self.path)
            try:
                if url.path == "/status":
                    return self._send(200, queue.status())
                if url.path.startswith("/log/"):
                    tail = int(parse_qs(url.query).get("tail", ["200"])[0])
                    return self._send(200, queue.log(url.path[5:], tail))
                if files_root and url.path == "/files":
                    rel = parse_qs(url.query).get("dir", [""])[0]
                    return self._send(200, {"files": list_files(files_root, rel)})
                if files_root and url.path.startswith("/files/"):
                    return self._send_file(resolve_file(files_root, unquote(url.path[7:])))
            except PermissionError:
                return self._send(403, {"error": "outside the shared folder"})
            except FileNotFoundError:
                return self._send(404, {"error": "not found"})
            self._send(404, {"error": "unknown path"})

        def do_POST(self):  # noqa: N802
            if not self._auth():
                return
            url = urlparse(self.path)
            try:
                if url.path == "/jobs":
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    return self._send(201, {"added": queue.add(body["name"], body["script"])})
                if url.path.startswith("/cancel/"):
                    return self._send(200, {"result": queue.cancel(url.path[8:])})
            except (ValueError, KeyError) as e:
                return self._send(400, {"error": str(e)})
            except FileExistsError:
                return self._send(409, {"error": "a job with that name exists"})
            except FileNotFoundError:
                return self._send(404, {"error": "no such job"})
            self._send(404, {"error": "unknown path"})

        def log_message(self, *args):  # keep the pod log readable
            pass

    return Handler


def serve(queue_dir: str | Path, port: int, token: str, files_root: str | Path | None = None,
          poll_s: float = 10.0) -> None:
    if len(token) < 32:
        raise SystemExit("use a token of at least 32 characters")
    queue = Queue(queue_dir)
    # Jobs left in running/ by a crash or a pod restart go back to the front of the queue.
    for name in queue.list("running"):
        shutil.move(queue.root / "running" / name, queue.root / "pending" / name)
    root = Path(files_root) if files_root else None
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(queue, token, root))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"worker on :{port}, queue {queue.root}", flush=True)
    while True:
        if queue.run_next() is None:
            time.sleep(poll_s)


def derive_token(secret: str, pod_name: str) -> str:
    """Worker token for a pod, derived from a secret both sides already hold.

    Nothing extra has to be stored: any later session that has the secret
    (the RunPod API key) can recompute the token of a pod from its name.
    """
    return hmac.new(secret.encode(), f"priism-worker:{pod_name}".encode(), "sha256").hexdigest()


# The RunPod HTTPS proxy (Cloudflare) answers 403 to urllib's default User-Agent.
USER_AGENT = "priism-worker-client/1"


def download(base_url: str, token: str, rel: str, dest: str | Path) -> Path:
    """Fetch one file from the worker, resuming a partial ``dest.part`` if present."""
    import urllib.request
    from urllib.parse import quote

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(base_url.rstrip("/") + "/files/" + quote(rel), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, part.open("ab" if r.status == 206 else "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.rename(dest)
    return dest


def client(base_url: str, token: str, method: str, path: str, body: dict | None = None) -> str:
    """Call a worker's HTTP API; returns the response body."""
    import urllib.request

    req = urllib.request.Request(base_url.rstrip("/") + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                                          "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()
