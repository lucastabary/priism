"""Factor heads: each slot says what it plays (Lucas's idea, 2026-10-09).

Every slot of the separator describes its source with three factors:

- **Z, identity**: one vector per slot for the whole excerpt (attention pooling over time and bands).
  Trained without instrument labels: the Z of the first half of the excerpt must find the Z of the second
  half of the same source among all other sources of the batch (contrastive), so Z keeps what stays the
  same over time, the timbre. Parts that one instrument plays (twins) count as the same identity: two
  identical 303s share a Z and differ by what they play. Classifying tracks later works on Z.
- **P, what it plays**: per frame, a piano-roll over the 128 MIDI pitches (several at once for chords),
  plus an onset and an "active" flag; unpitched sources (drums, noise) have an empty roll and only onsets.
- **V, variation**: a small vector per frame for everything else (how the note is played, the effects).
  Its time mean predicts the source's effect settings (delay, reverb, drive, filters, pan...) and each
  frame predicts the note velocity, so V is the place for expression and effects.

The truth comes from the generator (``timeline`` and ``fx`` in each source's metadata). Songs without it
(older sets, real songs) only train the separation.

The confidence of each slot is the existing existence probability. With ``feedback``, the predicted
roll is drawn back onto the slot's features as a harmonic comb over the bands (zero-initialised, so a run
resumed from a model without factors starts exactly where it was): the mask can follow the notes its slot
claims, the cue that tells two identical instruments apart.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_PITCH = 128
FX_KEYS = ("delay_send", "delay_feedback", "reverb_send", "reverb_rt60_s", "drive", "lowpass_hz", "highpass_hz",
           "pan", "width", "sidechain", "sweep_oct")


def fx_vector(fx: dict | None) -> np.ndarray:
    """Effect settings of one source, each scaled to about [0, 1] (a send's settings count only if it is on)."""
    if not fx:
        return np.full(len(FX_KEYS), np.nan, np.float32)
    d, r = fx.get("delay_send", 0.0), fx.get("reverb_send", 0.0)
    v = [d / 0.6, fx.get("delay_feedback", 0.0) / 0.8 * (d > 0), r / 0.3, fx.get("reverb_rt60_s", 0.0) / 4.0 * (r > 0),
         fx.get("drive", 0.0), math.log(fx.get("lowpass_hz", 20000.0) / 1500) / math.log(20000 / 1500),
         math.log(fx.get("highpass_hz", 20.0) / 20) / math.log(20), (fx.get("pan", 0.0) + 1) / 2,
         fx.get("width", 0.0) / 0.4, fx.get("sidechain", 0.0), fx.get("sweep_oct", 0.0) / 5]
    return np.asarray(v, np.float32)


def rasterize(timeline: np.ndarray, n_frames: int, frame_s: float) -> dict[str, torch.Tensor]:
    """One source's notes (E, 5: start s, end s, pitch or -1, velocity, onset) -> frame targets.

    Frame f is centred at ``f * frame_s`` (centred STFT). roll (T, 128), onset (T,) on the frame nearest
    each attack and the next one, active (T,), velocity (T,) of the loudest note sounding.
    """
    roll = torch.zeros(n_frames, N_PITCH)
    onset, active, vel = torch.zeros(n_frames), torch.zeros(n_frames), torch.zeros(n_frames)
    for a, b, p, v, o in np.asarray(timeline, np.float64).reshape(-1, 5):
        f0, f1 = max(0, int(math.ceil(a / frame_s - 0.5))), min(n_frames, int(math.ceil(b / frame_s - 0.5)))
        f1 = max(f1, f0 + 1) if f0 < n_frames else f1
        if f1 <= 0 or f0 >= n_frames:
            continue
        active[f0:f1] = 1.0
        vel[f0:f1] = torch.clamp(vel[f0:f1], min=float(v))
        if p >= 0:
            roll[f0:f1, int(round(min(p, N_PITCH - 1)))] = 1.0
        if o and a >= -frame_s / 2:
            fo = int(round(a / frame_s))
            onset[max(fo, 0):min(fo + 2, n_frames)] = 1.0
    return {"roll": roll, "onset": onset, "active": active, "vel": vel}


def harmonic_comb(band_hz: torch.Tensor, n_harm: int = 16) -> torch.Tensor:
    """(128, Nb): share of each pitch's harmonic energy (1/h) that falls in each band [lo, hi) Hz."""
    out = torch.zeros(N_PITCH, len(band_hz))
    for m in range(N_PITCH):
        f0 = 440.0 * 2 ** ((m - 69) / 12)
        for h in range(1, n_harm + 1):
            inside = (band_hz[:, 0] <= h * f0) & (h * f0 < band_hz[:, 1])
            out[m] += inside.float() / h
    return out / out.sum(1, keepdim=True).clamp_min(1e-6)


class FactorHeads(nn.Module):
    """Z / P / V of every slot from the slot's own features (N = slots, T frames, Nb bands, D)."""

    def __init__(self, dim: int, band_hz: torch.Tensor, width: int = 16, hidden: int = 256, z_dim: int = 64,
                 v_dim: int = 16, feedback: bool = False):
        super().__init__()
        nb = len(band_hz)
        self.squeeze = nn.Linear(dim, width)  # per (frame, band) token, then all bands of a frame together
        self.frame = nn.Sequential(nn.LayerNorm(nb * width), nn.Linear(nb * width, hidden), nn.GELU())
        self.time = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden // 16)  # a little context in time
        self.pitch = nn.Linear(hidden, N_PITCH)
        self.events = nn.Linear(hidden, 2)  # onset, active
        self.v = nn.Linear(hidden, v_dim)
        self.vel = nn.Linear(v_dim, 1)
        self.fx = nn.Linear(v_dim, len(FX_KEYS))
        self.score = nn.Linear(dim, 1)
        self.z = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, z_dim))
        self.feedback = feedback
        if feedback:
            self.register_buffer("comb", harmonic_comb(band_hz), persistent=False)
            self.back = nn.Linear(3, dim)
            nn.init.zeros_(self.back.weight)
            nn.init.zeros_(self.back.bias)

    def _pool(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:  # (N, t, Nb, D), (N, t, Nb) -> (N, z)
        w = torch.softmax(s.flatten(1).float(), dim=1).to(x.dtype)
        return F.normalize(self.z(torch.einsum("nt,ntd->nd", w, x.flatten(1, 2))).float(), dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        N, T, Nb, D = x.shape
        h = self.frame(self.squeeze(x).flatten(2))  # (N, T, hidden)
        h = h + self.time(h.transpose(1, 2)).transpose(1, 2)
        ev = self.events(h)
        v = self.v(h)
        s = self.score(x).squeeze(-1)
        half = T // 2
        z_halves = torch.stack([self._pool(x[:, :half], s[:, :half]), self._pool(x[:, half:], s[:, half:])], 1)
        return {"pitch": self.pitch(h).float(), "onset": ev[..., 0].float(), "active": ev[..., 1].float(),
                "v": v.float(), "vel": self.vel(v).squeeze(-1).float(), "fx": self.fx(v.mean(1)).float(),
                "z": self._pool(x, s), "z_halves": z_halves}

    def back_features(self, f: dict[str, torch.Tensor], dtype) -> torch.Tensor:
        """(N, T, Nb, D) to add to the slot features: the claimed notes' harmonics per band, onsets, activity."""
        comb = torch.sigmoid(f["pitch"]) @ self.comb  # (N, T, Nb)
        t = torch.stack([comb, torch.sigmoid(f["onset"])[..., None].expand_as(comb),
                         torch.sigmoid(f["active"])[..., None].expand_as(comb)], -1)
        return self.back(t.to(self.back.weight.dtype)).to(dtype)


def factor_loss(f: dict[str, torch.Tensor], labels: list[dict | None], match: list[tuple], K: int,
                frame_s: float, temp: float = 0.1) -> tuple[torch.Tensor, dict]:
    """Factor losses on the slots PIT matched to a source with a known timeline.

    f: FactorHeads output for B*K slots; labels: per item ``{"timeline": [arrays], "families": [...],
    "fx": (N, 11)}`` or None; match: per item (slot rows, target cols) from the PIT assignment.
    """
    dev = f["pitch"].device
    T = f["pitch"].shape[1]
    idx, tg, fam, fxs = [], [], [], []
    for b, (lab, (rows, cols)) in enumerate(zip(labels, match)):
        if not lab or lab.get("timeline") is None:
            continue
        for r, c in zip(rows, cols):
            tl = lab["timeline"][int(c)]
            if tl is None:
                continue
            idx.append(b * K + int(r))
            tg.append(rasterize(tl, T, frame_s))
            fam.append((b, lab["families"][int(c)]))
            fxs.append(lab["fx"][int(c)])
    if not idx:
        return torch.zeros((), device=dev), {}
    i = torch.as_tensor(idx, device=dev)
    roll = torch.stack([t["roll"] for t in tg]).to(dev)
    onset, active, vel = (torch.stack([t[k] for t in tg]).to(dev) for k in ("onset", "active", "vel"))
    pitched = (roll.sum((1, 2)) > 0).float()  # unpitched sources: the roll must stay empty, lighter weight
    lp = F.binary_cross_entropy_with_logits(f["pitch"][i], roll, pos_weight=torch.tensor(5.0, device=dev),
                                            reduction="none").mean((1, 2))
    lp = (lp * (0.25 + 0.75 * pitched)).mean()
    lo = F.binary_cross_entropy_with_logits(f["onset"][i], onset, pos_weight=torch.tensor(5.0, device=dev))
    la = F.binary_cross_entropy_with_logits(f["active"][i], active)
    lv = ((f["vel"][i] - vel).pow(2) * active).sum() / active.sum().clamp_min(1)
    fx = torch.as_tensor(np.stack(fxs), device=dev)
    ok = ~torch.isnan(fx)
    lfx = (f["fx"][i] - torch.nan_to_num(fx)).pow(2)[ok].mean() if ok.any() else torch.zeros((), device=dev)
    # Z: the second half of a source finds the first half's Z (and its twins') among the batch's sources.
    half = T // 2
    heard = (active[:, :half].mean(1) > 0.05) & (active[:, half:].mean(1) > 0.05)
    lz, z_acc = torch.zeros((), device=dev), float("nan")
    if int(heard.sum()) >= 2:
        za, zb = f["z_halves"][i][heard, 0], f["z_halves"][i][heard, 1]
        keys = [x for x, h in zip(fam, heard.tolist()) if h]
        pos = torch.tensor([[a == b for b in keys] for a in keys], device=dev, dtype=torch.float32)
        logits = za @ zb.T / temp
        logp = logits.log_softmax(1)
        lz = 0.5 * (-(logp * pos).sum(1) / pos.sum(1)).mean() + \
            0.5 * (-(logits.T.log_softmax(1) * pos.T).sum(1) / pos.T.sum(1)).mean()
        z_acc = float(pos.gather(1, logits.argmax(1, keepdim=True)).mean())
    loss = lp + lo + la + lv + lfx + 0.5 * lz
    with torch.no_grad():
        pred = (f["pitch"][i] > 0).float()
        sel = pitched > 0
        tp = (pred * roll)[sel].sum()
        f1 = float(2 * tp / (pred[sel].sum() + roll[sel].sum()).clamp_min(1)) if sel.any() else float("nan")
        on_pred = (f["onset"][i] > 0).float()
        on_f1 = float(2 * (on_pred * onset).sum() / (on_pred.sum() + onset.sum()).clamp_min(1))
    return loss, {"f_loss": float(loss.detach()), "pitch_f1": f1, "onset_f1": on_f1, "z_acc": z_acc,
                  "fx_mse": float(lfx.detach())}
