"""MUSDB18-HQ from Zenodo into FLAC stems on the volume, with the zip kept on the pod's local disk only.

The pod's network volume holds 50 GB in all: the zip plus its extracted wav files (~45 GB at the peak)
filled it on 2026-10-09 and stopped the queue worker. Here the 23 GB zip goes to the container disk
(``--tmp``, 45 GB free), downloaded as 64 MB ranges over 12 connections (Zenodo caps each connection
at a few MB/s; reading the zip remotely in small pieces ran at under 1 MB/s), written in place in a
preallocated file (no parts to concatenate, so no second copy). Each stem is then re-encoded to FLAC
(lossless) on the volume and the mixture skipped (it is the sum of the stems): ~15 GB there. The zip is
deleted at the end. Resumes: finished ranges (listed in <zip>.done) and finished songs are skipped.

Usage: python3 fetch_musdb_flac.py OUT_DIR [--tmp /root]
"""

from __future__ import annotations

import argparse
import http.client
import io
import os
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
TOTAL = 22656664047
UA = "priism-fetch"
CHUNK = 64 << 20


def fetch_range(url: str, path: Path, a: int, b: int) -> None:
    """Bytes a..b (inclusive) of ``url`` written at offset a of ``path``, retried until complete."""
    for attempt in range(10):
        got = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "r+b") as f:
                f.seek(a)
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
            if got == b - a + 1:
                return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(min(2 ** attempt, 60))
    raise OSError(f"range {a}-{b} failed")


def download(zip_path: Path, threads: int = 12) -> None:
    done_file = zip_path.with_suffix(".done")
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    if not zip_path.exists() or zip_path.stat().st_size != TOTAL:
        with open(zip_path, "wb") as f:
            f.truncate(TOTAL)  # sparse: blocks are only used as ranges arrive
        done = set()
        done_file.unlink(missing_ok=True)
    ranges = [(a, min(a + CHUNK, TOTAL) - 1) for a in range(0, TOTAL, CHUNK)]
    todo = [r for r in ranges if str(r[0]) not in done]
    lock, t0, n0 = threading.Lock(), time.time(), len(todo)
    print(f"{len(todo)} of {len(ranges)} ranges to fetch", flush=True)

    def one(r):
        fetch_range(URL, zip_path, *r)
        with lock:
            with open(done_file, "a") as f:
                f.write(f"{r[0]}\n")
            done.add(str(r[0]))
            left = sum(1 for x in ranges if str(x[0]) not in done)
            if left % 20 == 0:
                rate = (n0 - left) * CHUNK / max(time.time() - t0, 1) / 1e6
                print(f"{left} ranges left, {rate:.0f} MB/s", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, todo))


def extract(zip_path: Path, out: Path) -> None:
    z = zipfile.ZipFile(zip_path)
    members = [m for m in z.infolist() if m.filename.endswith(".wav") and not m.filename.endswith("mixture.wav")]
    print(f"{len(members)} stem files", flush=True)
    for i, m in enumerate(members):
        rel = Path(*Path(m.filename).parts[-3:])  # train|test / song / stem.wav
        dest = (out / rel).with_suffix(".flac")
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        x, sr = sf.read(io.BytesIO(z.read(m)), dtype="int16", always_2d=True)
        tmp = dest.with_suffix(".tmp")  # not an audio suffix: a crash leaves nothing training would read
        sf.write(tmp, x, sr, format="FLAC", subtype="PCM_16")
        tmp.rename(dest)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(members)} {rel}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tmp", default="/root")
    a = p.parse_args()
    zip_path = Path(a.tmp) / "musdb18hq.zip"
    download(zip_path)
    extract(zip_path, Path(a.out))
    os.remove(zip_path)
    zip_path.with_suffix(".done").unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
