"""Hand-made baselines under cleave-train's own identity protocol (id_pairs on mdb-dev2, seed 0, 8 chunks,
20 s gap, 200 tracks max): the bar Z must clear, measured on the same pairs as id_auc_kind."""
import sys, json, numpy as np, torch
from scipy.fft import dct
from priism.lab.cleave.data import find_songs, read_crop
from priism.lab.cleave.features import Frontend, HOP, SILENCE_DB, SR
from priism.lab.cleave.model import CleaveConfig
from priism.lab.cleave.train import id_pairs, _auc

torch.set_num_threads(8)
songs = find_songs([sys.argv[1]])
cfg = CleaveConfig()
crop = 8 * cfg.chunk_frames * HOP
pairs = id_pairs(songs, crop, int(20.0 * SR), 200, 0)
print("tracks", len(pairs), flush=True)
front = Frontend()
feats = {"logmel_mean": [], "logmel_meanstd": [], "mfcc20_meanstd": []}
for s, st, a, b, kind in pairs:
    w = torch.from_numpy(np.stack([read_crop(songs[s], st, x, crop) for x in (a, b)]))
    logmel, _, level = front(w)
    for i in range(2):
        m = logmel[i][level[i] > SILENCE_DB].numpy()
        if len(m) < 10:
            m = logmel[i].numpy()
        mf = dct(m, type=2, norm="ortho", axis=1)[:, :20]
        feats["logmel_mean"].append(m.mean(0))
        feats["logmel_meanstd"].append(np.concatenate([m.mean(0), m.std(0)]))
        feats["mfcc20_meanstd"].append(np.concatenate([mf.mean(0), mf.std(0)]))
kinds = np.array([p[4] for p in pairs])
n = len(pairs)
out = {}
rng = np.random.default_rng(0)
for name, f in feats.items():
    X = np.stack(f).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)  # standardised over the val crops (as a cosine-on-z-scores probe)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    za, zb = X[0::2], X[1::2]
    sim = za @ zb.T
    eye = np.eye(n, dtype=bool)
    same = kinds[:, None] == kinds[None, :]
    auc_k = _auc(sim[eye], sim[~eye & same])
    boots = []
    for _ in range(500):  # tracks resampled with replacement (pairs stay paired)
        j = rng.integers(0, n, n)
        s2, k2 = sim[np.ix_(j, j)], kinds[j]
        e2 = j[:, None] == j[None, :]
        boots.append(_auc(np.diag(s2), s2[~e2 & (k2[:, None] == k2[None, :])]))
    out[name] = {"id_auc_kind": auc_k, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                 "id_auc": _auc(sim[eye], sim[~eye])}
    print(name, out[name], flush=True)
json.dump(out, open("idbase.json", "w"), indent=1)
