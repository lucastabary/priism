"""Do the predictor's conditioning factors (P, L, onset) leak timbre? (docs/cleave-diagnostic.md 9.3)

The same note sequence is rendered with many random generator patches of one kind. A query is segment A of
patch i; its candidates are segment B of every patch, all playing the same notes. AUC = P(sim(iA, iB) >
sim(iA, jB)): 0.5 means the descriptor cannot tell patches apart (no timbre in it), 1.0 means it is a timbre
fingerprint. MFCC mean + std is the reference for "carries timbre".

Similarity is the cosine of z-scored summary statistics, as in idbase.py: a lower bound on what a trained
predictor could read from the same curves (a leak found here is real; no leak here is not a proof).

Usage: PYTHONPATH=src python scripts/cleave_diagnostic/pl_leak.py [patches per kind] [sequences per kind]
"""
import json
import sys

import numpy as np
import torch
from scipy.fft import dct

from priism.lab.cleave.features import N_PITCH, SILENCE_DB, SR, Frontend
from priism.lab.cleave.train import _auc
from priism.lab.gen.tonal import Note, render_notes, sample_patch

torch.set_num_threads(4)
N_PATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N_SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SEG_STEPS, BPM = 64, 120.0  # 8 s per segment
RANGES = {"bass": (31, 50), "lead": (57, 84), "pluck": (52, 80), "stab": (50, 74), "pad": (48, 72), "arp": (57, 84)}
rng = np.random.default_rng(0)
f_sal, f_bp = Frontend("salience"), Frontend("bp")


def sequence(lo, hi, chords):
    """Two segments of 16th-step notes (A then B), rests included; stabs and pads play triads."""
    notes, t, p = [], 0, (lo + hi) // 2
    while t < 2 * SEG_STEPS:
        length = int(rng.choice([1, 2, 4, 8] if chords else [1, 1, 2, 2, 4]))
        if rng.random() > 0.2:
            p = int(np.clip(p + rng.integers(-5, 6), lo, hi - 7))
            for q in ([p, p + 4, p + 7] if chords else [p]):
                notes.append(Note(float(t), float(min(length, 2 * SEG_STEPS - t)), float(q), float(rng.uniform(0.6, 1))))
        t += length
    return notes


def quantiles(x, k=10):
    return np.quantile(x, np.linspace(0.05, 0.95, k)) if len(x) else np.zeros(k)


def modspec(x, bands=16):
    """Modulation spectrum of a curve: |FFT| pooled in log bands, normalised (rhythm and envelope shape)."""
    m = np.abs(np.fft.rfft(x - x.mean()))[1:]
    edges = np.unique(np.geomspace(1, len(m), bands + 1).astype(int))
    v = np.array([m[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    return v / (v.sum() + 1e-9)


def velocity_roll(notes01, loud):
    """Lucas's float pitch: each played note carries the loudness at its onset, held for the note."""
    out = np.zeros_like(notes01)
    for k in range(notes01.shape[1]):
        on = notes01[:, k] > 0
        if not on.any():
            continue
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        for s in starts:
            e = s + np.argmin(on[s:]) if not on[s:].all() else len(on)
            out[s:e, k] = loud[s]
    return out


def describe(wave):
    w = torch.from_numpy(wave)[None]
    logmel, pl_s, level = f_sal(w)
    _, pl_b, _ = f_bp(w)
    act = (level[0] > SILENCE_DB).numpy()
    m = logmel[0].numpy()[act] if act.sum() > 10 else logmel[0].numpy()
    mf = dct(m, type=2, norm="ortho", axis=1)[:, :20]
    sal, loud, onset = pl_s[0, :, :N_PITCH].numpy(), pl_s[0, :, N_PITCH].numpy(), pl_s[0, :, N_PITCH + 1].numpy()
    bp = pl_b[0, :, :N_PITCH].numpy()
    vel = velocity_roll(bp, loud)
    d = {
        "mfcc_meanstd (timbre ref)": np.r_[mf.mean(0), mf.std(0)],
        "loudness": np.r_[quantiles(loud[act]), quantiles(np.diff(loud)), modspec(loud)],
        "onset (flux)": np.r_[quantiles(onset[act]), modspec(onset)],
        "pitch salience top-4": np.r_[sal.mean(0), sal.std(0)],
        "pitch basic-pitch 0/1": np.r_[bp.mean(0), bp.std(0)],
        "notes x loudness(t)": np.r_[(bp * loud[:, None]).mean(0), quantiles((bp * loud[:, None])[bp > 0])],
        "notes x velocity (held)": np.r_[vel.mean(0), quantiles(vel[bp > 0])],
    }
    # coarser candidates: onset events without heights, loudness smoothed over ~250 ms and cut to 4 levels
    peaks = (onset > np.r_[onset[1:], 0]) & (onset >= np.r_[0, onset[:-1]]) & (onset > 0.3)
    d["onset events 0/1"] = np.r_[quantiles(np.diff(np.flatnonzero(peaks)) if peaks.sum() > 1 else np.zeros(0)),
                                  peaks.mean(), modspec(peaks.astype(float))]
    smooth = np.convolve(loud, np.ones(21) / 21, mode="same")
    lq = np.digitize(smooth, [-0.75, -0.5, -0.25]) / 3.0  # loud is level_db / 80 in [-1, 0]
    d["loudness 4 levels, smoothed"] = np.r_[np.bincount((lq * 3).astype(int), minlength=4) / len(lq), modspec(lq)]
    d["bp + onset events"] = np.r_[d["pitch basic-pitch 0/1"], d["onset events 0/1"]]
    d["bp + onset events + loud 4 levels"] = np.r_[d["bp + onset events"], d["loudness 4 levels, smoothed"]]
    d["P/L today (sal+loud+onset)"] = np.r_[d["pitch salience top-4"], d["loudness"], d["onset (flux)"]]
    d["bp + loud + onset (running arm)"] = np.r_[d["pitch basic-pitch 0/1"], d["loudness"], d["onset (flux)"]]
    return d


def norm(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


seg = int(SEG_STEPS * 60 / BPM / 4 * SR)
groups = []  # one per (kind, sequence): {descriptor: (A features, B features)}
for kind, (lo, hi) in RANGES.items():
    for q in range(N_SEQ):
        notes = sequence(lo, hi, kind in ("stab", "pad"))
        feats = {}
        for i in range(N_PATCH):
            y = render_notes(notes, sample_patch(kind, rng), 2 * seg, SR, BPM, rng).astype(np.float32)
            y = 0.5 * y / (np.abs(y).max() + 1e-9)
            for half, x in (("A", y[:seg]), ("B", y[seg:])):
                for name, v in describe(x).items():
                    feats.setdefault(name, {"A": [], "B": []})[half].append(v)
        groups.append((kind, feats))
        print(f"{kind} seq {q} done", flush=True)

names = list(groups[0][1])
out = {}
for name in names:
    pos, neg, per_kind = [], [], {}
    # z-score each descriptor over all segments of all groups, then compare within a group only
    allX = np.concatenate([np.r_[np.stack(f[name]["A"]), np.stack(f[name]["B"])] for _, f in groups])
    mu, sd = allX.mean(0), allX.std(0) + 1e-8
    for kind, f in groups:
        A = (np.stack(f[name]["A"]) - mu) / sd
        B = (np.stack(f[name]["B"]) - mu) / sd
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
        sim = A @ B.T
        eye = np.eye(len(A), dtype=bool)
        pos.append(sim[eye]); neg.append(sim[~eye])
        per_kind.setdefault(kind, ([], []))
        per_kind[kind][0].append(sim[eye]); per_kind[kind][1].append(sim[~eye])
    # bootstrap over groups (kind x sequence) for a CI
    aucs = [_auc(p, n) for p, n in zip(pos, neg)]
    boots = [np.mean(np.array(aucs)[rng.integers(0, len(aucs), len(aucs))]) for _ in range(2000)]
    out[name] = {"auc": float(np.mean(aucs)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                 "per_kind": {k: _auc(np.concatenate(v[0]), np.concatenate(v[1])) for k, v in per_kind.items()}}

print(f"\n{'descriptor':34s} {'AUC':>6s}  {'95% CI':>15s}   " + " ".join(f"{k:>6s}" for k in RANGES))
for name, r in out.items():
    print(f"{name:34s} {r['auc']:6.3f}  [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]   "
          + " ".join(f"{r['per_kind'][k]:6.3f}" for k in RANGES))
json.dump(out, open("pl_leak.json", "w"), indent=1)
