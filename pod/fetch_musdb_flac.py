"""MUSDB18-HQ straight from Zenodo into FLAC stems, without storing the 23 GB zip.

The pod's network volume holds 50 GB in all: the zip plus its extracted wav files (~45 GB at the peak)
filled it on 2026-10-09 and stopped the queue worker. Here the zip is read in place over HTTP range
requests (its central directory first, then one member at a time), each stem is re-encoded to FLAC
(lossless, ~60 % of the wav size) and the mixture is skipped (it is the sum of the stems): ~11 GB on disk.
Resumes: songs already written are skipped.

Usage: python3 fetch_musdb_flac.py OUT_DIR
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf

URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
UA = "priism-fetch"


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests (with retries)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": UA,
                                                                "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == n:
                    self.pos += n
                    return data
            except OSError:
                pass
            time.sleep(2 ** attempt)
        raise OSError(f"range {self.pos}+{n} failed")

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main(out: str) -> None:
    out = Path(out)
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=8 << 20))
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
        if i % 20 == 0:
            print(f"{i + 1}/{len(members)} {rel}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
