"""Fetch MoisesDB straight from its zip URL, stems re-encoded to FLAC, without storing the 89 GB zip.

The zip is read in place with HTTP range requests (its central directory sits at the end), so each member
is downloaded once, decompressed on the fly and written as FLAC (lossless; silent stretches of stems shrink
to almost nothing). Non-audio members (metadata json) are copied as they are. Resumable: a member whose
output already exists is skipped. Stops cleanly when free disk space falls under ``--min-free-gb``.

Usage: python fetch_moisesdb.py URL_FILE OUT_DIR [--workers 4] [--min-free-gb 15]
The URL (signed, personal, expires) is read from URL_FILE so it never lands in logs or in the repository.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "priism-fetch/1.0"}


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests, with a read-ahead buffer."""

    def __init__(self, url: str, size: int, chunk: int = 16 << 20):
        self.url, self.size, self.chunk = url, size, chunk
        self.pos, self.buf_start, self.buf = 0, 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, a: int, b: int) -> bytes:
        err: Exception | None = None
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == b - a + 1:
                    return data
            except Exception as e:  # network hiccup: retry with backoff
                err = e
            time.sleep(2 ** attempt)
        raise OSError(f"range {a}-{b} failed: {err or 'short read'}")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.buf_start + len(self.buf)
        if not (self.buf_start <= self.pos and self.pos + n <= end):
            a = self.pos
            b = min(self.size, a + max(n, self.chunk)) - 1
            self.buf_start, self.buf = a, self._fetch(a, b)
        i = self.pos - self.buf_start
        out = self.buf[i:i + n]
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url_file")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N audio members (for a test)")
    args = ap.parse_args()

    import soundfile as sf

    url = Path(args.url_file).read_text().strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = remote_size(url)
    names = [i for i in zipfile.ZipFile(RangeFile(url, size, chunk=1 << 20)).infolist() if not i.is_dir()]
    audio = [i for i in names if i.filename.lower().endswith(".wav")]
    other = [i for i in names if not i.filename.lower().endswith(".wav")]
    if args.limit:
        audio = audio[:args.limit]
    print(f"zip {size / 1e9:.1f} GB: {len(audio)} wav, {len(other)} other members", flush=True)

    stop = threading.Event()
    done = [0, 0]  # members written, bytes written
    lock = threading.Lock()

    def job(info: zipfile.ZipInfo) -> None:
        if stop.is_set():
            return
        is_wav = info.filename.lower().endswith(".wav")
        dest = out / (info.filename[:-4] + ".flac" if is_wav else info.filename)
        if dest.exists():
            return
        if shutil.disk_usage(out).free < args.min_free_gb * 1e9:
            if not stop.is_set():
                print(f"free disk under {args.min_free_gb} GB: stopping", flush=True)
            stop.set()
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(RangeFile(url, size))
        with z.open(info) as src:
            if not is_wav:
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f, 4 << 20)
                tmp.rename(dest)
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=out) as tw:
                    shutil.copyfileobj(src, tw, 4 << 20)
                    tw.flush()
                    data, sr = sf.read(tw.name, dtype="int32", always_2d=True)
                    sub = sf.info(tw.name).subtype
                tmp = dest.with_name(dest.name + ".part")
                sf.write(tmp, data, sr, format="FLAC", subtype="PCM_24" if "24" in sub else "PCM_16")
                tmp.rename(dest)
        with lock:
            done[0] += 1
            done[1] += dest.stat().st_size
            if done[0] % 50 == 0:
                print(f"{done[0]} written, {done[1] / 1e9:.1f} GB", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(job, other + audio))
    print(f"finished: {done[0]} new files, {done[1] / 1e9:.1f} GB" + (" (stopped on disk space)" if stop.is_set() else ""))
    sys.exit(2 if stop.is_set() else 0)


if __name__ == "__main__":
    main()
