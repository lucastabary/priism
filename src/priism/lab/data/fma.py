"""Fetch chosen genres of the Free Music Archive (FMA) without downloading the whole archive.

FMA (https://github.com/mdeff/fma) is 106,574 Creative Commons tracks with a
genre taxonomy. Its audio ships as single zip files (93 GiB of 30 s clips,
879 GiB of full tracks), but the host serves byte ranges, so we read the zip's
central directory remotely and pull only the members we want. These are the
real, unlabelled songs for MixIT training and listening tests.

Most FMA licenses are non-commercial (CC BY-NC*): fine for research, to be
reconsidered if Priism is ever distributed. Each fetch writes a manifest with
the license of every track.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable

BASE_URL = "https://os.unil.cloud.switch.ch/fma"
USER_AGENT = "priism/0.1 (research)"

# Genres worth fetching for Priism (FMA titles); 'Electronic' alone is ~34k tracks.
DEFAULT_GENRES = ["Techno", "House", "Minimal Electronic", "Dubstep", "Drum & Bass", "Jungle", "Reggae - Dub",
                  "Breakbeat", "Dance", "Garage", "Downtempo", "Trip-Hop", "IDM", "Glitch"]

csv.field_size_limit(sys.maxsize)


def _http_range(url: str) -> Callable[[int, int], bytes]:
    def fetch(start: int, end: int) -> bytes:  # inclusive end, like the Range header
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    return fetch


def _http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


class RangeFile(io.RawIOBase):
    """Read-only seekable file over a range fetcher, with a small block cache.

    ``zipfile`` only needs seek/tell/read, so a remote zip opens without
    downloading it; each member read costs a few range requests.
    """

    def __init__(self, fetch: Callable[[int, int], bytes], size: int, block: int = 1 << 20, cache_blocks: int = 64):
        super().__init__()
        self._fetch, self._size, self._block = fetch, size, block
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._cache_blocks = cache_blocks
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def _get_block(self, i: int) -> bytes:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        start = i * self._block
        data = self._fetch(start, min(start + self._block, self._size) - 1)
        self._cache[i] = data
        if len(self._cache) > self._cache_blocks:
            self._cache.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        end = self._size if n is None or n < 0 else min(self._size, self._pos + n)
        if end - self._pos > 4 * self._block:  # big member body: one direct request, not cached
            data = self._fetch(self._pos, end - 1)
        else:
            parts = []
            p = self._pos
            while p < end:
                b = p // self._block
                blk = self._get_block(b)
                off = p - b * self._block
                take = min(len(blk) - off, end - p)
                parts.append(blk[off:off + take])
                p += take
            data = b"".join(parts)
        self._pos += len(data)
        return data

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(RangeFile(_http_range(url), _http_size(url)))


# ---------------------------------------------------------------------------------------------- metadata

def ensure_metadata(meta_dir: str | Path) -> Path:
    """tracks.csv and genres.csv, read straight out of the remote metadata zip if not on disk."""
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    need = [n for n in ("tracks.csv", "genres.csv") if not (meta_dir / n).exists()]
    if need:
        with open_remote_zip(f"{BASE_URL}/fma_metadata.zip") as z:
            for n in need:
                (meta_dir / n).write_bytes(z.read(f"fma_metadata/{n}"))
    return meta_dir


def load_genres(meta_dir: Path) -> dict[str, int]:
    with open(meta_dir / "genres.csv", newline="") as f:
        return {r["title"]: int(r["genre_id"]) for r in csv.DictReader(f)}


def load_tracks(meta_dir: Path) -> list[dict]:
    """One dict per track: id, title, artist, genres_all, genre_top, license, duration, subset."""
    with open(meta_dir / "tracks.csv", newline="") as f:
        rd = csv.reader(f)
        top, sub, _ = next(rd), next(rd), next(rd)
        col = {(a, b): i for i, (a, b) in enumerate(zip(top, sub))}
        want = {"title": ("track", "title"), "artist": ("artist", "name"), "genres_all": ("track", "genres_all"),
                "genre_top": ("track", "genre_top"), "license": ("track", "license"),
                "duration": ("track", "duration"), "subset": ("set", "subset")}
        out = []
        for row in rd:
            if not row or not row[0].isdigit():
                continue
            d = {k: row[col[c]] for k, c in want.items()}
            d["id"] = int(row[0])
            d["genres_all"] = json.loads(d["genres_all"] or "[]")
            d["duration"] = int(float(d["duration"] or 0))
            out.append(d)
        return out


def select_tracks(tracks: list[dict], genre_ids: dict[str, int], genres: list[str], limit_per_genre: int | None = None,
                  min_duration: int = 60) -> list[dict]:
    """Tracks carrying any of ``genres`` (anywhere in their genre tree), balanced across genres."""
    unknown = [g for g in genres if g not in genre_ids]
    if unknown:
        raise KeyError(f"unknown FMA genres: {unknown}")
    picked: dict[int, dict] = {}
    for g in genres:
        gid = genre_ids[g]
        hits = [t for t in tracks if gid in t["genres_all"] and t["duration"] >= min_duration and t["id"] not in picked]
        hits.sort(key=lambda t: (t["id"] * 2654435761) % 2**32)  # deterministic shuffle
        for t in hits[:limit_per_genre]:
            picked[t["id"]] = {**t, "picked_for": g}
    return sorted(picked.values(), key=lambda t: t["id"])


def member_name(track_id: int, subset: str) -> str:
    tid = f"{track_id:06d}"
    return f"fma_{subset}/{tid[:3]}/{tid}.mp3"


def fetch(out_dir: str | Path, genres: list[str] | None = None, subset: str = "full", limit_per_genre: int | None = 200,
          meta_dir: str | Path | None = None, max_gb: float | None = None, min_free_gb: float = 10.0,
          log=print) -> Path:
    """Download the chosen tracks to ``out_dir/<id>.mp3`` and write ``manifest.csv``. Resumable.

    Stops early once the folder holds ``max_gb`` or the disk has less than ``min_free_gb`` free
    (the pod volume is shared with other projects).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = ensure_metadata(meta_dir or out / "_metadata")
    chosen = select_tracks(load_tracks(meta), load_genres(meta), genres or DEFAULT_GENRES, limit_per_genre)
    log(f"{len(chosen)} tracks selected")
    used = sum(f.stat().st_size for f in out.glob("*.mp3"))
    with open_remote_zip(f"{BASE_URL}/fma_{subset}.zip") as z:
        names = set(z.namelist())
        for i, t in enumerate(chosen):
            if (max_gb and used > max_gb * 1e9) or shutil.disk_usage(out).free < min_free_gb * 1e9:
                log(f"stopping at {used / 1e9:.1f} GB (size cap or low disk)")
                break
            dst = out / f"{t['id']:06d}.mp3"
            name = member_name(t["id"], subset)
            if dst.exists() or name not in names:
                t["file"] = dst.name if dst.exists() else ""
                continue
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(z.read(name))
            tmp.rename(dst)
            used += dst.stat().st_size
            t["file"] = dst.name
            if (i + 1) % 25 == 0:
                log(f"{i + 1}/{len(chosen)}")
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "file", "picked_for", "genre_top", "title", "artist", "duration", "license"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(t for t in chosen if t.get("file"))
    return out
