"""Training songs generated while the model trains, so nothing has to be stored or waited for.

Songs are fully determined by their seed, so keeping them is pointless: a few
generator processes keep a rolling pool of recent songs on local disk (oldest
deleted), and ``LiveSongs`` serves random crops of whatever the pool holds.
Every song is seen for a while then replaced, so a long run never loops over
the same set.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .data import crop_example

TRAIN_SEED_BASE = 10**10  # far from the fixed validation seeds (9e8...)


def ready_songs(folder: str | Path) -> list[Path]:
    """Songs whose meta.json exists (write_song writes it last), oldest first."""
    metas = [p for p in Path(folder).glob("song_*/meta.json")]
    metas.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return [p.parent for p in metas]


def _generate_forever(folder: str, worker: int, workers: int, first: int, duration_s: float, sample_rate: int,
                      max_songs: int, sources: tuple[int, int] | None = None) -> None:
    from ..gen.song import write_song

    os.nice(5)  # the data loader and the training process come first
    k = 0
    while True:
        seed = first + worker + k * workers
        k += 1
        try:
            n = sources[0] + seed % (sources[1] - sources[0] + 1) if sources else None
            write_song(seed, folder, duration_s=duration_s, sample_rate=sample_rate, n_sources=n)
        except Exception as e:  # a bad seed must not stop the stream
            print(f"song {seed} failed: {e!r}", flush=True)
            shutil.rmtree(Path(folder) / f"song_{seed:08d}", ignore_errors=True)
            continue
        if worker == 0:
            songs = ready_songs(folder)
            for old in songs[: max(0, len(songs) - max_songs)]:
                shutil.rmtree(old, ignore_errors=True)


class SongStream:
    """``workers`` processes writing songs into ``folder`` until ``stop()``; keeps about ``max_songs``.

    ``sources=(lo, hi)`` limits the number of sources per song (curriculum: few sources first).
    """

    def __init__(self, folder: str | Path, workers: int, duration_s: float = 30.0, sample_rate: int = 44100,
                 max_songs: int = 400, first_seed: int | None = None, sources: tuple[int, int] | None = None):
        self.folder = Path(folder)
        if self.folder.exists():  # songs of an earlier run may follow another curriculum
            shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True, exist_ok=True)
        # A new seed range on each start (a resumed run gets fresh songs).
        first = first_seed if first_seed is not None else TRAIN_SEED_BASE + int(time.time()) * 1000
        ctx = mp.get_context("spawn")
        self.procs = [ctx.Process(target=_generate_forever, daemon=True,
                                  args=(str(self.folder), w, workers, first, duration_s, sample_rate, max_songs,
                                        sources))
                      for w in range(workers)]
        for p in self.procs:
            p.start()

    def wait(self, min_songs: int, timeout_s: float = 3600, log=print) -> None:
        t0 = time.time()
        while len(ready_songs(self.folder)) < min_songs:
            if not any(p.is_alive() for p in self.procs):
                raise RuntimeError("song generators died")
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"fewer than {min_songs} songs after {timeout_s} s")
            time.sleep(2)
        log(f"{len(ready_songs(self.folder))} songs ready after {time.time() - t0:.0f} s")

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.join(timeout=10)


class LiveSongs(IterableDataset):
    """Endless random crops of the songs currently in ``folder`` (re-listed every ``refresh`` crops)."""

    def __init__(self, folder: str | Path, chunk_s: float, sample_rate: int, seed: int = 0, lossy_p: float = 0.0,
                 refresh: int = 32, labels: bool = False):
        self.folder = Path(folder)
        self.chunk = int(chunk_s * sample_rate)
        self.sr = sample_rate
        self.seed = seed
        self.lossy_p = lossy_p
        self.refresh = refresh
        self.labels = labels

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng((self.seed, info.id if info else 0, os.getpid(), time.time_ns()))
        songs: list[Path] = []
        n = 0
        while True:
            if n % self.refresh == 0 or not songs:
                songs = ready_songs(self.folder)
                if not songs:
                    time.sleep(1)
                    continue
            n += 1
            song = songs[int(rng.integers(len(songs)))]
            try:  # the pool may delete a song between listing and reading
                meta = json.loads((song / "meta.json").read_text())
                yield crop_example(song, meta, self.chunk, self.sr, rng, self.lossy_p, with_labels=self.labels)
            except (FileNotFoundError, RuntimeError, OSError):
                songs = []
