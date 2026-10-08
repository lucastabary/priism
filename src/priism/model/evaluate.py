"""Score the class-agnostic separator on real songs with grouped stems (NI .stem.mp4: 4 stems each).

The model splits finer than the reference (one track per instrument, the stems
are groups such as "Drums" or "Synths"), so its outputs are first grouped: each
output, the rest included, goes to the reference stem it explains best (largest
projection). The grouped estimates are then scored with SNR, and against the
mix itself (SNR improvement, SNRi), which needs no other model to compare to.

The grouping is an oracle (it looks at the reference), so even an untrained
model gets a positive SNRi: compare runs with each other and with the step-0
model, not with zero.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .loss import neg_snr


def read_stem_mp4(path: str | Path, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """(mix (2, S), stems (4, 2, S)) from a NI stem file: stream 0 is the mix, streams 1-4 the stems."""
    streams = []
    for i in range(5):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{i}", "-f", "f32le",
                              "-ac", "2", "-ar", str(sample_rate), "-"], capture_output=True, check=True).stdout
        streams.append(np.frombuffer(raw, np.float32).reshape(-1, 2).T)
    n = min(s.shape[1] for s in streams)
    streams = [s[:, :n] for s in streams]
    stems = np.stack(streams[1:])
    return stems.sum(0), stems  # the stems' sum, so the reference is exactly additive


def group_outputs(est: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """est (K, C, S) outputs, refs (J, C, S) -> (J, C, S): each output added to its best-explained stem."""
    e = est.flatten(1)
    r = refs.flatten(1)
    proj = (e @ r.T) / (r.pow(2).sum(1) + 1e-8)  # (K, J): share of each stem in each output
    best = proj.argmax(1)
    out = torch.zeros_like(refs)
    out.index_add_(0, best, est)
    return out


@torch.no_grad()
def score_song(model, mix: np.ndarray, stems: np.ndarray, sr: int, segment_s: float, segments: int,
               device: str, threshold: float = 0.5) -> list[dict]:
    """Scores of ``segments`` evenly spread excerpts of one song."""
    seg = int(segment_s * sr)
    total = mix.shape[1]
    starts = np.linspace(0.1 * total, 0.9 * total - seg, segments).astype(int) if total > 2 * seg else [0]
    rows = []
    for st in starts:
        m = torch.from_numpy(np.ascontiguousarray(mix[:, st:st + seg])).to(device)
        r = torch.from_numpy(np.ascontiguousarray(stems[:, :, st:st + seg])).to(device)
        active = r.pow(2).mean((1, 2)) > 1e-6 * m.pow(2).mean()  # silent stems cannot be scored
        out = model(m[None])
        keep = torch.sigmoid(out["exist_logits"][0].float()) > threshold
        src = out["sources"][0].float()[keep]
        est = torch.cat([src, (m - src.sum(0))[None]])  # the rest makes the sum exact
        grouped = group_outputs(est, r)
        snr = -neg_snr(grouped, r, soft_db=100.0)
        base = -neg_snr(m[None].expand_as(r), r, soft_db=100.0)
        rows.append({"start_s": round(st / sr, 1), "n_tracks": int(keep.sum()),
                     "snr": [float(x) if a else None for x, a in zip(snr, active)],
                     "snri": [float(x - b) if a else None for x, b, a in zip(snr, base, active)]})
    return rows


def evaluate_folder(run: str | Path, folder: str | Path, out: str | Path, segment_s: float = 8.0, segments: int = 3,
                    device: str = "cpu", msst_path: str | Path | None = None, limit: int | None = None,
                    log=print) -> dict:
    from .train import load_run

    model, sr = load_run(run, device, msst_path)
    folder, out = Path(folder), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = {}
    index = folder / "INDEX.tsv"
    if index.exists():
        with index.open() as f:
            names = {r["fichier"]: [s.strip() for s in r["stems"].split(",")] for r in csv.DictReader(f, delimiter="\t")}
    files = sorted(folder.glob("*.stem.mp4"))[:limit]
    per_song = {}
    for f in files:
        mix, stems = read_stem_mp4(f, sr)
        per_song[f.name] = {"stems": names.get(f.name), "segments": score_song(model, mix, stems, sr, segment_s,
                                                                               segments, device)}
        vals = [x for s in per_song[f.name]["segments"] for x in s["snri"] if x is not None]
        log(f"{f.name}: SNRi {np.mean(vals):.2f} dB" if vals else f"{f.name}: no active stem")
    all_snri = [x for s in per_song.values() for g in s["segments"] for x in g["snri"] if x is not None]
    all_snr = [x for s in per_song.values() for g in s["segments"] for x in g["snr"] if x is not None]
    summary = {"run": str(run), "songs": len(files), "segment_s": segment_s,
               "mean_snr": float(np.mean(all_snr)) if all_snr else None,
               "mean_snri": float(np.mean(all_snri)) if all_snri else None,
               "median_snri": float(np.median(all_snri)) if all_snri else None}
    (out / "per_song.json").write_text(json.dumps(per_song, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))
    return summary
