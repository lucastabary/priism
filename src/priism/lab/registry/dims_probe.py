"""How much does compressing the trunk tokens cost? Raw-token tracking at several PCA sizes, no training.

The cache keeps 32 PCA dimensions of the 256 the BS-Roformer trunk gives per token (75 % of the variance).
This probe fits a full PCA (all 256 directions) on training songs of a multitrack set (MoisesDB: train
artists only), then for every dev2 and test song runs the trunk once and scores whole-song clustering on the
first d directions, for each d and each cut: track-level V-measure per (song, d, cut), written as JSON lines.
Only scores are stored, never the 256-dim tokens (they would take ~150 MB per song). The cut is then tuned per
d on dev2 and reported on test (``summarize``).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .features import Projection, band_energy, pool_time, read_multitrack, song_tokens
from .metrics import ownership, paired_compare, score
from .run import in_part, list_songs

DIMS = (32, 64, 128, 256)
CUTS = (0.1, 0.2, 0.3, 0.4, 0.5)


def _key(song) -> str:
    return (song.labels[0].get("group_key") if song.labels else None) or song.name


def _score_song(args) -> list[dict]:
    """Worker: one song's tokens (T, Nb, 256) from a temporary file -> one row per (d, cut)."""
    path, name, part, chunk = args
    with np.load(path) as d:  # closed before the unlink (Windows keeps open files)
        tokens, energy = d["tokens"].astype(np.float32), d["energy"].astype(np.float32)
    Path(path).unlink()
    from .track import whole_song

    _, audible = ownership(energy)
    rows = []
    for dims in DIMS:
        for cut in CUTS:
            r = score(whole_song(tokens[..., :dims], audible, cut), energy, chunk)
            rows.append({"song": name, "part": part, "dims": dims, "cut": cut, "v_measure": r["v_measure"],
                         "n_clusters": r["n_clusters"]})
    return rows


def run(trunk, folder: str | Path, out: str | Path, tmp: str | Path, fit_songs: int = 16, pool: int = 4,
        jobs: int = 4, log=print) -> None:
    out, tmp = Path(out), Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    sr = trunk.sample_rate
    paths = list_songs("multitrack", folder)
    # 1. Full PCA on training artists only: no dev2 or test song shapes the features.
    proj, n = Projection(), 0
    for p in paths:
        song = read_multitrack(p, sr)
        if n >= fit_songs or not in_part(_key(song), "train"):
            continue
        h = song_tokens(trunk, song.mix)
        e = band_energy(trunk, song.mix)[: len(h)]
        proj.update(h, e > e.max() * 1e-6)
        n += 1
    mean = proj._s / proj._n
    eig = np.sort(np.linalg.eigvalsh(proj._ss / proj._n - np.outer(mean, mean)))[::-1]
    proj.fit(256)
    log(f"full PCA on {n} train songs; variance kept by 32/64/128 dims: "
        + ", ".join(f"{eig[:d].sum() / eig.sum():.0%}" for d in (32, 64, 128)))
    # 2. Dev2 and test songs: the trunk runs here (GPU), the clustering in worker processes.
    done = {json.loads(line)["song"] for line in out.read_text().splitlines()} if out.exists() else set()
    chunk = max(1, round(4.0 / (trunk.hop * pool / sr)))
    with ProcessPoolExecutor(jobs) as ex, out.open("a") as f:
        pending = []

        def drain(block: bool) -> None:  # write finished songs; keep at most 2 x jobs temporary files
            while pending and (block or pending[0].done() or len(pending) >= 2 * jobs):
                for row in pending.pop(0).result():
                    f.write(json.dumps(row) + "\n")
                f.flush()

        for p in paths:
            song = read_multitrack(p, sr)
            key = _key(song)
            part = "test" if in_part(key, "test") else "dev2" if in_part(key, "dev2") else None
            if part is None or song.name in done:
                continue
            tokens = pool_time(proj(song_tokens(trunk, song.mix)), pool).astype(np.float16)
            energy = np.stack([pool_time(band_energy(trunk, x), pool, reduce="sum") for x in song.sources])
            path = tmp / f"{song.name}.npz"
            np.savez(path, tokens=tokens, energy=energy.astype(np.float32))
            pending.append(ex.submit(_score_song, (str(path), song.name, part, chunk)))
            drain(False)
            log(f"{part}/{song.name}")
        drain(True)


def summarize(path: str | Path, artists: dict[str, str] | None = None) -> str:
    """Cut tuned per dims on dev2 (mean V), then test mean V (and per-artist mean if ``artists``), with the
    paired gain of each size over 32 dims."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    dims_list, cuts = sorted({r["dims"] for r in rows}), sorted({r["cut"] for r in rows})
    lines, test_v = [], {}
    for dims in dims_list:
        dev = {c: np.mean([r["v_measure"] for r in rows if r["dims"] == dims and r["cut"] == c and r["part"] == "dev2"])
               for c in cuts}
        cut = max(dev, key=dev.get)
        t = {r["song"]: r["v_measure"] for r in rows if r["dims"] == dims and r["cut"] == cut and r["part"] == "test"}
        test_v[dims] = t
        lines.append(f"{dims} dims: cut {cut}, test V {np.mean(list(t.values())):.3f} ({len(t)} songs)")
    base = test_v[dims_list[0]]
    for dims in dims_list[1:]:
        common = sorted(set(base) & set(test_v[dims]))
        c = paired_compare([test_v[dims][s] for s in common], [base[s] for s in common])
        line = f"{dims} - {dims_list[0]}: {c['mean_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] d_z {c['d_z']:.2f}"
        if artists:
            per = {}
            for s in common:
                per.setdefault(artists[s], []).append(test_v[dims][s] - base[s])
            line += f" | artist-balanced {np.mean([np.mean(v) for v in per.values()]):+.3f} ({len(per)} artists)"
        lines.append(line)
    return "\n".join(lines)
