"""Score a separator per instrument on real multitracks (one file per instrument), e.g. mshoxxDB.

NI stems only have 4 groups (Drums, Bass, Synths...), so they cannot tell whether two synths end up
in separate tracks. Here every instrument has its own reference, and two scores are given:

- ``strict``: one output per instrument (Hungarian matching on SNR), the identity score. Two
  instruments merged in one output leave one of them unmatched, at 0 dB or less.
- ``grouped``: each output joins the instrument it explains best (oracle grouping, as in
  ``eval-sep``), lenient: a source split over several outputs is not punished.

Instruments listed in ``group_names`` (default: drums) are references made of several sources;
for them, both scores use the grouped estimate, since a drum stem is many instruments.

Folder layout: one folder per song with ``<song>.flac`` (mix, unused: the sum of the stems is the
reference, so it is exactly additive) and ``<song>_<name>.flac`` per instrument (mshoxxDB).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

# model_fn(mix (2, S)) -> estimates (K, 2, S): kept outputs, plus the rest if the model has one
ModelFn = Callable[[np.ndarray], np.ndarray]


def snr(ref: np.ndarray, est: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(ref**2) / (np.sum((ref - est) ** 2) + 1e-12) + 1e-12))


def read_song(folder: Path, sr: int = 44100) -> tuple[list[str], np.ndarray]:
    """(instrument names, stems (J, 2, S)) of one song folder; mono stems are copied to both channels."""
    names, stems = [], []
    for f in sorted(folder.glob(f"{folder.name}_*.flac")):
        x, fsr = sf.read(f, dtype="float32", always_2d=True)
        if fsr != sr:
            raise ValueError(f"{f}: {fsr} Hz, expected {sr}")
        x = x.T
        stems.append(np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2])
        names.append(f.stem[len(folder.name) + 1:])
    n = min(s.shape[1] for s in stems)
    return names, np.stack([s[:, :n] for s in stems])


def score_excerpt(est: np.ndarray, refs: np.ndarray, groups: np.ndarray) -> tuple[list[float], list[float]]:
    """est (K, C, S), refs (J, C, S), groups (J,) bool -> (strict SNR per ref, grouped SNR per ref)."""
    e, r = est.reshape(len(est), -1), refs.reshape(len(refs), -1)
    # Each output joins the instrument holding most of its energy (length of its projection). eval-sep
    # divides by the stem power instead, which with many quiet stems sends loud outputs to quiet ones.
    best = ((e @ r.T) / (np.sqrt(np.sum(r**2, 1)) + 1e-8)).argmax(1)
    grouped = np.zeros_like(refs)
    for k, j in enumerate(best):
        grouped[j] += est[k]
    g = [snr(refs[j], grouped[j]) for j in range(len(refs))]
    # Strict: group references (drums) take their grouped outputs; others get one output each.
    free = [k for k in range(len(est)) if not groups[best[k]]]
    solo = [j for j in range(len(refs)) if not groups[j]]
    s = list(g)
    for j in solo:
        s[j] = snr(refs[j], np.zeros_like(refs[j]))  # unmatched: no output at all
    if free and solo:
        m = np.array([[snr(refs[j], est[k]) for k in free] for j in solo])
        rows, cols = linear_sum_assignment(-m)
        for a, b in zip(rows, cols):
            s[solo[a]] = float(m[a, b])
    return s, g


def active_mask(refs: np.ndarray, level: float) -> np.ndarray:
    """Instruments playing in an excerpt (J, C, S): at most 30 dB under ``level``, the mean power of the
    whole song's mix. Relative to the song, not the excerpt: some stems carry a constant noise floor,
    which would count as playing in a quiet outro."""
    return np.mean(refs**2, axis=(1, 2)) > 1e-3 * level


def busiest(stems: np.ndarray, seg: int, segments: int, level: float) -> list[int]:
    """Starts of ``segments`` non-overlapping excerpts with the most instruments playing (intros are
    often two parts, which says little about a full arrangement); ties go to the earliest."""
    cands = list(range(0, max(1, stems.shape[2] - seg + 1), seg // 2))
    n = [int(active_mask(stems[:, :, c:c + seg], level).sum()) for c in cands]
    starts: list[int] = []
    for i in sorted(range(len(cands)), key=lambda i: -n[i]):
        if all(abs(cands[i] - st) >= seg for st in starts):
            starts.append(cands[i])
        if len(starts) == segments:
            break
    return sorted(starts)


def evaluate(model_fn: ModelFn, root: str | Path, out: str | Path, segment_s: float = 8.0, segments: int = 3,
             sr: int = 44100, group_names: tuple[str, ...] = ("drums",), limit: int | None = None,
             log=print) -> dict:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    songs = sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob(f"{p.name}_*.flac")))[:limit]
    seg = int(segment_s * sr)
    rows = []
    for song in songs:
        names, stems = read_song(song, sr)
        groups = np.array([any(g in n.lower() for g in group_names) for n in names])
        level = float(np.mean(stems.sum(0) ** 2))
        starts = busiest(stems, seg, segments, level)
        for st in starts:
            refs = stems[:, :, st:st + seg]
            mix = refs.sum(0)
            active = active_mask(refs, level)
            if active.sum() < 2:
                continue
            est = model_fn(mix)
            s, g = score_excerpt(est, refs[active], groups[active])
            for name, a, b, grp in zip(np.array(names)[active], s, g, groups[active]):
                rows.append({"song": song.name, "start_s": round(st / sr, 1), "stem": str(name), "group": bool(grp),
                             "strict": a, "grouped": b, "n_out": len(est), "n_ref": int(active.sum())})
        done = [r for r in rows if r["song"] == song.name and not r["group"]]
        if done:
            log(f"{song.name}: strict {np.mean([r['strict'] for r in done]):.1f} dB, "
                f"grouped {np.mean([r['grouped'] for r in done]):.1f} dB (instruments, drums aside)")
    solo = [r for r in rows if not r["group"]]
    summary = {"songs": len(songs), "excerpts": len({(r['song'], r['start_s']) for r in rows}),
               "strict_mean": float(np.mean([r["strict"] for r in solo])),
               "strict_above_5db": float(np.mean([r["strict"] > 5 for r in solo])),
               "grouped_mean": float(np.mean([r["grouped"] for r in solo])),
               "grouped_above_5db": float(np.mean([r["grouped"] > 5 for r in solo])),
               "drums_mean": float(np.mean([r["grouped"] for r in rows if r["group"]] or [np.nan]))}
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))
    return summary


def separator_fn(run, device: str = "cpu", msst_path=None, threshold: float = 0.5) -> ModelFn:
    """A train-sep run as a ``ModelFn``: kept outputs plus the rest (the sum is the mix)."""
    import torch

    from .train import load_run

    model, _ = load_run(run, device, msst_path)

    @torch.no_grad()
    def fn(mix: np.ndarray) -> np.ndarray:
        m = torch.from_numpy(np.ascontiguousarray(mix))[None].to(device)
        o = model(m)
        keep = torch.sigmoid(o["exist_logits"][0].float()) > threshold
        src = o["sources"][0].float()[keep]
        return torch.cat([src, (m[0] - src.sum(0))[None]]).cpu().numpy()

    return fn


def msst_fn(config, ckpt, msst_path, device: str = "cpu") -> ModelFn:
    """A pretrained MSST model with fixed stems (e.g. BS-Roformer-SW) as a ``ModelFn``, for reference."""
    import torch

    from .msst_core import load_msst_roformer

    model = load_msst_roformer(config, ckpt, msst_path).to(device).eval()

    @torch.no_grad()
    def fn(mix: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(np.ascontiguousarray(mix))[None].to(device))[0].float().cpu().numpy()

    return fn


def cascade_fn(first: ModelFn, second: ModelFn, quiet_db: float = -40.0) -> ModelFn:
    """``first`` (e.g. BS-Roformer-SW, fixed stems) on the mix, then ``second`` (a class-free run) on each of
    its stems; stems more than ``quiet_db`` under the mix are kept whole. Outputs still sum to the mix."""

    def fn(mix: np.ndarray) -> np.ndarray:
        level = float(np.mean(mix**2)) + 1e-12
        out = []
        for stem in first(mix):
            loud = 10 * np.log10(float(np.mean(stem**2)) / level + 1e-12) > quiet_db
            out.extend(second(np.ascontiguousarray(stem, dtype=np.float32)) if loud else [stem])
        return np.stack(out)

    return fn

