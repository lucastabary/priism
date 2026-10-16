"""How much of the Cleave prediction target can Z still explain once P/L and the patch position are given?

Targets: (a) the raw log-mel patch, (b) the layer-normed tokens of the random-init target encoder (the
`frozen` probe's targets, seed 0). Inputs: position (tcol, row) one-hot, P/L of the patch's 16 frames,
and an oracle Z = mean/std log-mel of ANOTHER crop of the same track (identity, not this crop's content).
MLP fit on 70 % of songs, R² on the other songs' patches; 3 seeds."""
import sys, json, numpy as np, torch
from priism.lab.cleave.data import find_songs, read_crop, active_frac, song_frames_sr
from priism.lab.cleave.features import Frontend, SR, SILENCE_DB
from priism.lab.cleave.model import Cleave, CleaveConfig
import torch.nn.functional as F

torch.set_num_threads(8)
songs = find_songs([sys.argv[1]])
cfg = CleaveConfig()
torch.manual_seed(0)
model = Cleave(cfg).eval()
front = Frontend()
rng = np.random.default_rng(0)
CROP = 5 * cfg.chunk_frames * 512  # 5 chunks, ~11 s
rows = []  # per patch: song, track, X_pos, X_pl, Z, y_mel, y_tok
for si, s in enumerate(songs):
    total = song_frames_sr(s)
    if total < 3 * CROP:
        continue
    for st in range(len(s.stems)):
        got = []
        for _ in range(12):
            a = int(rng.integers(0, total - CROP))
            x = read_crop(s, st, a, CROP)
            if active_frac(x) >= 0.8 and (not got or abs(a - got[0][0]) > 2 * CROP):
                got.append((a, x))
            if len(got) == 2:
                break
        if len(got) < 2:
            continue
        w = torch.from_numpy(np.stack([got[0][1], got[1][1]]))
        logmel, pl, level = front(w)
        lb = logmel[1][level[1] > SILENCE_DB]
        z = torch.cat([lb.mean(0), lb.std(0)]).numpy()
        patches, plc = model.chunk(logmel[:1], pl[:1])  # (1, C, P, 256), (1, C, ncol, 16*86)
        C = patches.shape[1]
        act = level[0, : C * cfg.chunk_frames].reshape(C, -1).amin(-1) > SILENCE_DB
        with torch.no_grad():
            _, t = model.target(patches[0])
            t = F.layer_norm(t.float(), (t.shape[-1],))
        for c in range(C):
            if not act[c]:
                continue
            for p in range(cfg.n_patches):
                tc, r = divmod(p, cfg.n_rows)
                rows.append((si, len(rows) and 0, tc, r, plc[0, c, tc].numpy(), z, patches[0, c, p].numpy(), t[c, p].numpy()))
    print("song", si, len(rows), flush=True)
song_ids = np.array([r[0] for r in rows])
pos = np.zeros((len(rows), cfg.n_tcol + cfg.n_rows), np.float32)
for i, r in enumerate(rows):
    pos[i, r[2]] = 1; pos[i, cfg.n_tcol + r[3]] = 1
PL = np.stack([r[4] for r in rows]).astype(np.float32)
Z = np.stack([r[5] for r in rows]).astype(np.float32)
YM = np.stack([r[6] for r in rows]).astype(np.float32)
YT = np.stack([r[7] for r in rows]).astype(np.float32)
print("patches", len(rows), "songs", len(np.unique(song_ids)))

def fit(X, Y, tr, te, seed, epochs=12):
    torch.manual_seed(seed)
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
    my, sy = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xt, Yt = torch.from_numpy((X - mx) / sx), torch.from_numpy((Y - my) / sy)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                              torch.nn.GELU(), torch.nn.Linear(512, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    idx = np.flatnonzero(tr)
    for ep in range(epochs):
        np.random.default_rng(seed + ep).shuffle(idx)
        for b in range(0, len(idx), 512):
            j = torch.from_numpy(idx[b:b + 512])
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = net(Xt[te]).numpy() * sy + my
    Yte = Y[te]
    return float(1 - ((P - Yte) ** 2).sum() / ((Yte - Y[tr].mean(0)) ** 2).sum())

conds = {"pos": [pos], "pos+PL": [pos, PL], "pos+Z": [pos, Z], "pos+PL+Z": [pos, PL, Z]}
res = {}
us = np.unique(song_ids)
for seed in range(3):
    perm = np.random.default_rng(100 + seed).permutation(us)
    te = np.isin(song_ids, perm[: len(us) * 3 // 10])
    tr = ~te
    for tname, Y in (("logmel", YM), ("rand_tokens", YT)):
        for cname, parts in conds.items():
            r2 = fit(np.concatenate(parts, 1), Y, tr, te, seed)
            res.setdefault(tname, {}).setdefault(cname, []).append(r2)
            print(seed, tname, cname, round(r2, 4), flush=True)
out = {t: {c: (float(np.mean(v)), float(np.std(v))) for c, v in d.items()} for t, d in res.items()}
for t, d in res.items():
    gain = np.array(d["pos+PL+Z"]) - np.array(d["pos+PL"])
    head = gain / (1 - np.array(d["pos+PL"]))
    out[t]["Z_gain_over_PL"] = (float(gain.mean()), float(gain.std()))
    out[t]["Z_share_of_residual"] = (float(head.mean()), float(head.std()))
print(json.dumps(out, indent=1))
json.dump(out, open("headroom.json", "w"), indent=1)
