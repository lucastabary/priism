"""Scale of the frozen probe's targets: per-element variance of the random-init target tokens around their
mean token (what a predictor that knows nothing still has to explain), on active dev2 stems."""
import sys, numpy as np, torch
import torch.nn.functional as F
from priism.lab.cleave.data import find_songs, read_crop, active_frac, song_frames_sr
from priism.lab.cleave.features import Frontend, SILENCE_DB
from priism.lab.cleave.model import Cleave, CleaveConfig
torch.set_num_threads(2)
songs = find_songs([sys.argv[1]])
cfg = CleaveConfig(); torch.manual_seed(0); model = Cleave(cfg).eval(); front = Frontend()
rng = np.random.default_rng(1); toks = []
CROP = 5 * cfg.chunk_frames * 512
for s in songs[:12]:
    for st in range(len(s.stems)):
        a = int(rng.integers(0, song_frames_sr(s) - CROP)); x = read_crop(s, st, a, CROP)
        if active_frac(x) < 0.8: continue
        logmel, pl, level = front(torch.from_numpy(x[None]))
        patches, _ = model.chunk(logmel, pl)
        with torch.no_grad():
            _, t = model.target(patches[0]); toks.append(F.layer_norm(t.float(), (t.shape[-1],)).reshape(-1, t.shape[-1]))
T = torch.cat(toks)
print("tokens", len(T), "MSE of predicting 0:", float((T**2).mean()), "of predicting the mean token:", float(((T - T.mean(0))**2).mean()))
