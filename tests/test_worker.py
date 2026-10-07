import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from priism.worker import Queue, make_handler

TOKEN = "t" * 40


def test_jobs_run_in_name_order_and_land_in_done_or_failed(tmp_path):
    q = Queue(tmp_path)
    q.add("0020-second.sh", "echo second >> order.txt\nexit 3\n")
    q.add("0010-first.sh", "echo first >> order.txt\n")
    assert q.run_next() == "0010-first.sh"
    assert q.run_next() == "0020-second.sh"
    assert q.run_next() is None
    assert (tmp_path / "order.txt").read_text().split() == ["first", "second"]
    assert q.list("done") == ["0010-first.sh"] and q.list("failed") == ["0020-second.sh"]
    assert "exit=3" in q.log("0020-second.sh")


def test_add_rejects_bad_and_duplicate_names(tmp_path):
    q = Queue(tmp_path)
    with pytest.raises(ValueError):
        q.add("../evil.sh", "")
    q.add("0010-a.sh", "true")
    with pytest.raises(FileExistsError):
        q.add("0010-a.sh", "true")


def _server(tmp_path):
    q = Queue(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(q, TOKEN))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return q, srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _call(url, token=TOKEN, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {token}"}, method="POST" if body else "GET")
    with urllib.request.urlopen(req) as r:
        return r.status, r.read().decode()


def test_http_api_needs_the_token_and_queues_jobs(tmp_path):
    q, srv, base = _server(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _call(base + "/status", token="wrong")
        assert e.value.code == 401
        code, _ = _call(base + "/jobs", body={"name": "0010-hello.sh", "script": "echo hello"})
        assert code == 201
        status = json.loads(_call(base + "/status")[1])
        assert status["pending"] == ["0010-hello.sh"]
        q.run_next()
        assert "hello" in _call(base + "/log/0010-hello.sh")[1]
    finally:
        srv.shutdown()


def test_files_can_be_listed_downloaded_and_resumed(tmp_path):
    from priism.worker import download

    root = tmp_path / "ws"
    (root / "runs").mkdir(parents=True)
    payload = bytes(range(256)) * 4000
    (root / "runs" / "model.ckpt").write_bytes(payload)
    q = Queue(tmp_path / "q")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(q, TOKEN, root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        listing = json.loads(_call(base + "/files?dir=runs")[1])["files"]
        assert listing == [{"path": "runs/model.ckpt", "size": len(payload), "mtime": listing[0]["mtime"]}]
        dest = tmp_path / "local" / "model.ckpt"
        dest.parent.mkdir()
        (tmp_path / "local" / "model.ckpt.part").write_bytes(payload[:1000])  # interrupted earlier
        download(base, TOKEN, "runs/model.ckpt", dest)
        assert dest.read_bytes() == payload
        with pytest.raises(urllib.error.HTTPError) as e:
            _call(base + "/files/..%2F..%2Fetc%2Fpasswd")
        assert e.value.code == 403
    finally:
        srv.shutdown()


def test_cpu_and_gpu_lanes_run_side_by_side_and_respect_after(tmp_path):
    import time

    q = Queue(tmp_path)
    q.add("0010-data.sh", "# lane: cpu\nsleep 1\necho data > data.txt\n")
    q.add("0020-more-data.sh", "# lane: cpu\necho more\n")
    q.add("0090-test.sh", "# after: 0010\ncat data.txt > seen.txt\n")
    q.add("0100-train.sh", "echo train\n")
    # The GPU lane may not start 0090 before 0010 is over, and 0100 keeps its place behind it.
    assert q.next_job("gpu") is None
    t = threading.Thread(target=q.run_next, args=("cpu",))
    t.start()
    time.sleep(0.3)
    assert q.list("running") == ["0010-data.sh"] and q.next_job("gpu") is None
    t.join()
    assert q.next_job("gpu") == "0090-test.sh"
    assert q.run_next("gpu") == "0090-test.sh"
    assert (tmp_path / "seen.txt").read_text().strip() == "data"
    assert q.run_next("cpu") == "0020-more-data.sh"
    assert q.run_next("gpu") == "0100-train.sh"
