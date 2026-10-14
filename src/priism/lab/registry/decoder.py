"""Fine decoder: split one BS-Roformer-SW stem into its sources with bin-resolution masks, one per query.

The registry (``separate.py``) splits a stem with token masks: one weight per 46 ms x band token, so two
sources sharing a band in a token cannot be told apart (``oracle`` in ``separate_eval`` measures that
limit). This decoder keeps the registry's queries (identity-head centroids) but predicts, per query, a mask
at the STFT's own resolution (frame x bin x channel), in the spirit of DANet (Chen et al. 2017: attractors =
source centroids in an embedding space, masks from them) and CLAPSep (FiLM by a query embedding on the
features of a frozen pretrained encoder):

1. the frozen SW trunk's taps (block 6 and final_norm, 2 x 256 per frame and band, at full frame rate) go
   through a linear to ``dim`` and get the stem's own band features (log power, share of the stems' power),
   so one decoder serves every stem;
2. per query: FiLM by (query, stem id), then ``depth`` small axial RoFormer blocks (attention over time
   within a band, then over bands within a frame, rotary positions), so a query follows its line in time;
3. a per-band MLP (BS-Roformer's MaskEstimator layout: one MLP per band, outputs the band's bins x
   channels) reads the block output and the stem's log magnitude at each bin of the band, and gives one
   logit per (frame, bin, channel);
4. the logits of the K queries of one stem go through a softmax over K, so the masks are a partition of
   unity: the sub-sources (stem estimate x mask) sum to the stem estimate exactly (iSTFT is linear), and
   the engine contract (stems + residual = mix) still holds.

The masks are real ratios on the complex SW estimate (no phase change): anything else breaks the exact sum.
SW's own mask heads are not reused: they map 256-dim final features to a complex mask of the stem *against
the mix*, a different question from *which query owns this bin*; their shapes do not match ``dim`` either.

Start point. The logits get a prior: cos(identity-head embedding of the token, query) / ``prior_temp``,
the registry's own soft partition, and the head's last layer starts at zero; so at step 0 the decoder is
exactly the registry's soft token split (temp ``prior_temp``) and training learns the bin-level residual.
The first validation is therefore the baseline to beat.

Training (``train_decoder``): random crops of synthetic songs (``priism-lab gen``), SW run online in bf16
under no_grad, each reference source assigned to the SW stem that holds most of its energy (each bin's
power shared among stems by their power share), queries = centroid of the head embeddings of the tokens
the source owns in the crop (``own``), loss = L1-SNR on the waveform + multi-resolution STFT magnitude L1
(log ratio) of each sub-source against its reference, plus a level term against silent outputs. Queries
come from ground truth at training and from the registry clusters at inference (whole-song centroids): a
known gap, narrowed by ``query_noise``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _log(*a) -> None:  # flushed: a redirected log shows progress as it happens
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------- the network

def _rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary positions (Su et al. 2021) along dim -2 of (..., L, dh)."""
    L, dh = x.shape[-2:]
    half = dh // 2
    freq = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / half))
    ang = torch.arange(L, device=x.device, dtype=torch.float32)[:, None] * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:2 * half]
    return torch.cat([a * cos - b * sin, a * sin + b * cos, x[..., 2 * half:]], -1)


class _Attn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, d)
        N, L, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(N, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(_rope(q), _rope(k), v)
        return x + self.out(y.transpose(1, 2).reshape(N, L, d))


class _FF(nn.Module):
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))

    def forward(self, x):
        return x + self.net(x)


class AxialBlock(nn.Module):
    """One RoFormer-style block of BS-Roformer's layout: time attention within each band, then band attention
    within each frame, each followed by a feed-forward."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 2):
        super().__init__()
        self.t_attn, self.t_ff = _Attn(dim, heads), _FF(dim, ff_mult)
        self.f_attn, self.f_ff = _Attn(dim, heads), _FF(dim, ff_mult)

    def _time(self, x):
        N, T, Nb, d = x.shape
        y = x.permute(0, 2, 1, 3).reshape(N * Nb, T, d)
        y = self.t_ff(self.t_attn(y))
        return y.reshape(N, Nb, T, d).permute(0, 2, 1, 3)

    def _band(self, x):
        N, T, Nb, d = x.shape
        return self.f_ff(self.f_attn(x.reshape(N * T, Nb, d))).reshape(N, T, Nb, d)

    def forward(self, x):  # (N, T, Nb, d)
        ck = (lambda f, a: checkpoint(f, a, use_reentrant=False)) if self.training and torch.is_grad_enabled() \
            else (lambda f, a: f(a))
        return ck(self._band, ck(self._time, x))


class BandMaskHead(nn.Module):
    """Per-band MLP (BS-Roformer's MaskEstimator layout), with bands of equal size batched in one einsum:
    (N, T, Nb, d) features + (N, T, F, C) stem log magnitude -> (N, T, F, C) logits. The output layer starts
    at zero (all logits 0: the prior alone decides at step 0)."""

    def __init__(self, dim: int, band_bins: np.ndarray, channels: int, hidden: int):
        super().__init__()
        sizes = band_bins[:, 1] - band_bins[:, 0]
        self.C = channels
        self.groups = []  # (size, band indices, bins in band order)
        perm = []
        for i, size in enumerate(sorted(set(sizes.tolist()))):
            bands = np.where(sizes == size)[0]
            bins = np.concatenate([np.arange(band_bins[b, 0], band_bins[b, 1]) for b in bands])
            self.register_buffer(f"bands{i}", torch.from_numpy(bands).long(), persistent=False)
            self.register_buffer(f"bins{i}", torch.from_numpy(bins).long(), persistent=False)
            n, m = len(bands), int(size) * channels
            setattr(self, f"w1_{i}", nn.Parameter(torch.randn(n, dim, hidden) * dim ** -0.5))
            setattr(self, f"wm_{i}", nn.Parameter(torch.randn(n, m, hidden) * m ** -0.5))
            setattr(self, f"b1_{i}", nn.Parameter(torch.zeros(n, hidden)))
            setattr(self, f"w2_{i}", nn.Parameter(torch.zeros(n, hidden, m)))
            setattr(self, f"b2_{i}", nn.Parameter(torch.zeros(n, m)))
            self.groups.append((int(size), len(bands)))
            perm.append(bins)
        inv = np.argsort(np.concatenate(perm))
        self.register_buffer("inv", torch.from_numpy(inv).long(), persistent=False)

    def forward(self, x: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        N, T = x.shape[:2]
        out = []
        for i, (size, n) in enumerate(self.groups):
            xb = x.index_select(2, getattr(self, f"bands{i}"))  # (N, T, n, d)
            mb = mag.index_select(2, getattr(self, f"bins{i}")).reshape(N, T, n, size * self.C)
            h = torch.einsum("ntbd,bdh->ntbh", xb, getattr(self, f"w1_{i}").to(xb.dtype))
            h = h + torch.einsum("ntbm,bmh->ntbh", mb.to(xb.dtype), getattr(self, f"wm_{i}").to(xb.dtype))
            h = F.gelu(h + getattr(self, f"b1_{i}").to(xb.dtype))
            o = torch.einsum("ntbh,bhm->ntbm", h, getattr(self, f"w2_{i}").to(h.dtype)) + getattr(self, f"b2_{i}").to(h.dtype)
            out.append(o.reshape(N, T, n * size, self.C))
        return torch.cat(out, 2).index_select(2, self.inv)


class FineDecoder(nn.Module):
    """See the module docstring. ``band_bins`` (Nb, 2) must cover the STFT bins [0, F) contiguously."""

    def __init__(self, tap_dim: int, band_bins, n_stems: int, query_dim: int, channels: int = 2, dim: int = 128,
                 depth: int = 1, heads: int = 4, hidden: int = 256, ff_mult: int = 2, prior_temp: float = 0.1,
                 pool: int = 4):
        super().__init__()
        band_bins = np.asarray(band_bins, np.int64)
        if band_bins[0, 0] != 0 or not (band_bins[1:, 0] == band_bins[:-1, 1]).all():
            raise ValueError("bands must cover the bins contiguously from 0")
        self.cfg = {"tap_dim": tap_dim, "band_bins": band_bins.tolist(), "n_stems": n_stems,
                    "query_dim": query_dim, "channels": channels, "dim": dim, "depth": depth, "heads": heads,
                    "hidden": hidden, "ff_mult": ff_mult, "prior_temp": prior_temp, "pool": pool}
        self.n_bins = int(band_bins[-1, 1])
        self.prior_temp, self.pool = prior_temp, pool
        Nb = len(band_bins)
        band_of_bin = np.repeat(np.arange(Nb), band_bins[:, 1] - band_bins[:, 0])
        self.register_buffer("band_of_bin", torch.from_numpy(band_of_bin).long(), persistent=False)
        self.inp = nn.Sequential(nn.LayerNorm(tap_dim), nn.Linear(tap_dim, dim))
        self.stem_in = nn.Linear(2, dim)
        self.band_pos = nn.Parameter(torch.zeros(Nb, dim))
        self.stem_emb = nn.Embedding(n_stems, 16)
        self.film = nn.Sequential(nn.Linear(query_dim + 16, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.blocks = nn.ModuleList(AxialBlock(dim, heads, ff_mult) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = BandMaskHead(dim, band_bins, channels, hidden)
        self.prior_scale = nn.Parameter(torch.ones(()))

    def _head(self, x, mag):
        return self.head(self.norm(x), mag)

    def forward(self, taps: torch.Tensor, item_crop: torch.Tensor, item_stem: torch.Tensor,
                stem_spec: torch.Tensor, stem_feat: torch.Tensor, queries: torch.Tensor, q_item: torch.Tensor,
                prior: torch.Tensor | None = None) -> torch.Tensor:
        """taps (B, T, Nb, Dt) per crop; items = (crop, stem) pairs: ``item_crop``/``item_stem`` (I,),
        ``stem_spec`` (I, C, F, T) complex SW estimates, ``stem_feat`` (I, T, Nb, 2) (``stem_features``);
        ``queries`` (Q, e) unit vectors, ``q_item`` (Q,) the item of each query; ``prior`` (B, T, Nb, e)
        unit head embeddings at frame rate or None -> masks (Q, C, F, T) float32, summing to 1 over the
        queries of each item."""
        I = len(item_crop)
        x0 = self.inp(taps)[item_crop] + self.band_pos + self.stem_in(stem_feat.to(taps.dtype))  # (I, T, Nb, d)
        cond = torch.cat([queries.to(x0.dtype), self.stem_emb(item_stem)[q_item].to(x0.dtype)], -1)
        gamma, beta = self.film(cond).chunk(2, -1)
        x = x0[q_item] * (1 + gamma[:, None, None]) + beta[:, None, None]  # (Q, T, Nb, d)
        for blk in self.blocks:
            x = blk(x)
        p = stem_spec.abs().pow(2)  # (I, C, F, T)
        mx = p.flatten(1).amax(1).clamp_min(1e-20)[:, None, None, None]
        mag = (torch.log10(p / mx + 1e-8) / 8).permute(0, 3, 2, 1)  # (I, T, F, C) in [-1, 0]
        if self.training and torch.is_grad_enabled():
            logits = checkpoint(self._head, x, mag[q_item], use_reentrant=False)
        else:
            logits = self._head(x, mag[q_item])
        logits = logits.float()  # (Q, T, F, C)
        if prior is not None:
            cos = torch.einsum("qtbe,qe->qtb", prior[item_crop][q_item].float(), queries.float())
            logits = logits + (self.prior_scale * cos / self.prior_temp)[..., self.band_of_bin][..., None]
        # softmax over the queries of each item: pad to (I, Kmax), -inf where an item has fewer queries
        q_pos = _positions(q_item)
        K = int(q_pos.max()) + 1
        pad = logits.new_full((I, K, *logits.shape[1:]), float("-inf"))
        pad[q_item, q_pos] = logits
        masks = torch.softmax(pad, 1)[q_item, q_pos]  # (Q, T, F, C)
        return masks.permute(0, 3, 2, 1)  # (Q, C, F, T)


def _positions(q_item: torch.Tensor) -> torch.Tensor:
    """Rank of each query among the queries of its item (queries of one item need not be contiguous)."""
    seen: dict[int, int] = {}
    pos = []
    for i in q_item.tolist():
        pos.append(seen.get(i, 0))
        seen[i] = pos[-1] + 1
    return torch.tensor(pos, device=q_item.device)


def stem_features(spec: torch.Tensor, band_bins: np.ndarray) -> torch.Tensor:
    """(N, C, F, T) complex stem estimates of one crop -> (N, T, Nb, 2): band log power of each stem relative
    to the crop's loudest stems' band (log10 / 8, floor -1) and the stem's share of the stems' power."""
    p = spec.abs().pow(2).sum(1)  # (N, F, T)
    c = torch.cat([torch.zeros_like(p[:, :1]), p.double().cumsum(1)], 1)
    lo = torch.as_tensor(band_bins[:, 0], device=p.device)
    hi = torch.as_tensor(band_bins[:, 1], device=p.device)
    bp = (c[:, hi] - c[:, lo]).float().clamp_min(0).transpose(1, 2)  # (N, T, Nb)
    tot = bp.sum(0, keepdim=True)
    lvl = (torch.log10(bp / tot.max().clamp_min(1e-20) + 1e-8) / 8).clamp_min(-1)
    return torch.stack([lvl, bp / tot.clamp_min(1e-20)], -1)


def load_decoder(path: str | Path, device: str = "cpu") -> FineDecoder:
    d = torch.load(path, map_location="cpu", weights_only=False)
    dec = FineDecoder(**d["cfg"])
    dec.load_state_dict(d["state"])
    return dec.to(device).eval()


def save_decoder(path: str | Path, dec: FineDecoder, **extra) -> None:
    torch.save({"cfg": dec.cfg, "state": {k: v.detach().cpu() for k, v in dec.state_dict().items()}, **extra}, path)


# ------------------------------------------------------------------------------------------ STFT helpers

def _stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    lead = x.shape[:-1]
    y = torch.stft(x.reshape(-1, x.shape[-1]), n_fft, hop, window=torch.hann_window(n_fft, device=x.device),
                   center=True, return_complex=True)
    return y.reshape(*lead, *y.shape[-2:])


def _istft(spec: torch.Tensor, n_fft: int, hop: int, length: int) -> torch.Tensor:
    """(..., F, T) complex -> (..., length): the SW STFT (hann, centred, not normalised)."""
    lead = spec.shape[:-2]
    y = torch.istft(spec.reshape(-1, *spec.shape[-2:]), n_fft, hop, window=torch.hann_window(n_fft, device=spec.device),
                    center=True, length=length)
    return y.reshape(*lead, length)


# ------------------------------------------------------------------------------------------ training data

def song_dirs(folders: list[str | Path]) -> list[Path]:
    """Generator output folders (or song folders) -> song folders (those with a meta.json)."""
    out = []
    for f in folders:
        f = Path(f)
        out += [f] if (f / "meta.json").exists() else sorted(p.parent for p in f.glob("*/meta.json"))
    return out


class SynthCrops(torch.utils.data.Dataset):
    """Random crops of generator songs, read straight from the source files (no whole song in memory).
    Item i of epoch-free sampling: ``fixed`` [(song, start)] pins the crops (validation)."""

    def __init__(self, songs: list[Path], sr: int, crop_s: float, seed: int = 0, length: int = 10 ** 9,
                 fixed: list[tuple[int, int]] | None = None):
        self.songs, self.sr, self.n = songs, sr, int(round(crop_s * sr))
        self.seed, self.length, self.fixed = seed, length, fixed
        self.meta = [json.loads((p / "meta.json").read_text()) for p in songs]
        self.frames = [sf.info(str(p / m["sources"][0]["file"])).frames for p, m in zip(songs, self.meta)]

    def __len__(self):
        return len(self.fixed) if self.fixed is not None else self.length

    def pick(self, rng: np.random.Generator) -> tuple[int, int]:
        si = int(rng.integers(len(self.songs)))
        return si, int(rng.integers(max(1, self.frames[si] - self.n)))

    def __getitem__(self, i: int) -> np.ndarray:
        si, start = self.fixed[i] if self.fixed is not None else self.pick(np.random.default_rng((self.seed, i)))
        out = []
        for s in self.meta[si]["sources"]:
            x, fsr = sf.read(str(self.songs[si] / s["file"]), start=start, frames=self.n, dtype="float32",
                             always_2d=True)
            if fsr != self.sr:
                raise ValueError(f"{self.songs[si]}: {fsr} Hz, expected {self.sr}")
            x = np.repeat(x.T, 2, 0) if x.shape[1] == 1 else x.T[:2]
            if x.shape[1] < self.n:
                x = np.pad(x, ((0, 0), (0, self.n - x.shape[1])))
            out.append(x)
        return np.stack(out)  # (J, C, n)


@dataclass
class Batch:
    taps: torch.Tensor  # (B, T, Nb, Dt)
    prior: torch.Tensor | None  # (B, T, Nb, e)
    item_crop: torch.Tensor  # (I,)
    item_stem: torch.Tensor
    stem_spec: torch.Tensor  # (I, C, F, T) complex64
    stem_feat: torch.Tensor  # (I, T, Nb, 2)
    queries: torch.Tensor  # (Q, e)
    q_item: torch.Tensor  # (Q,)
    refs: torch.Tensor  # (Q, C, S)
    ref_power: torch.Tensor  # (Q, C, F, T): the ideal ratio mask's numerators
    scale: torch.Tensor  # (Q,) rms of the crop's mix
    length: int

    def decode(self, decoder: FineDecoder, prior: bool = True) -> torch.Tensor:
        return decoder(self.taps, self.item_crop, self.item_stem, self.stem_spec, self.stem_feat, self.queries,
                       self.q_item, self.prior if prior else None)


def _pool(x: torch.Tensor, pool: int, reduce: str = "mean") -> torch.Tensor:
    """Groups of ``pool`` frames along dim 0 (the last group may be shorter), as ``features.pool_time``."""
    T = x.shape[0]
    n = -(-T // pool)
    pad = n * pool - T
    xs = torch.cat([x, x.new_zeros(pad, *x.shape[1:])]) if pad else x
    s = xs.reshape(n, pool, *x.shape[1:]).sum(1)
    if reduce == "mean":
        cnt = torch.full((n,), float(pool), device=x.device)
        cnt[-1] = pool - pad
        s = s / cnt.reshape(-1, *[1] * (x.dim() - 1)).to(s.dtype)
    return s


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def head_embeddings(taps: torch.Tensor, proj, head, pool: int) -> torch.Tensor:
    """(T, Nb, Dt) taps of one crop -> (Tp, Nb, e) unit head embeddings of its tokens, built as the cache
    builds tokens (PCA, mean over ``pool`` frames, float16) so the head sees its training distribution."""
    mean = torch.as_tensor(proj.mean, device=taps.device)
    basis = torch.as_tensor(proj.basis, device=taps.device)
    tok = _pool((taps.float() - mean) @ basis, pool).half().float()
    if head is None:
        return unit(tok)
    dev = next(head.parameters()).device
    return unit(head(tok.to(dev)).float().to(taps.device))


@torch.no_grad()
def prepare_batch(model, head, proj, crops: list[np.ndarray], pool: int = 4, own: float = 0.5,
                  present_db: float = -30.0, floor_db: float = 60.0, max_queries: int = 32,
                  query_noise: float = 0.0, rng: np.random.Generator | None = None) -> Batch | None:
    """Crops (J, C, S) of reference sources -> one ``Batch`` (None if no SW stem holds two present sources).

    Per crop: SW on the mix; each reference present (>= ``present_db`` under the mix) goes to the SW stem
    holding most of its energy (each bin's reference power shared among stems by their power share there);
    its query is the mean unit head embedding of the tokens it owns (share of the references' power >=
    ``own``, audible), or of all audible tokens weighted by its share squared when it owns fewer than 4.
    Only stems with >= 2 queries are items (one query: mask 1, nothing to learn). ``max_queries`` caps the
    queries per batch (items dropped at random), the main memory knob."""
    rng = rng or np.random.default_rng()
    dev = getattr(model, "device", "cpu")
    mixes = torch.from_numpy(np.stack([c.sum(0) for c in crops])).to(dev)
    S = mixes.shape[-1]
    taps, spec = model.taps_and_spec(mixes)  # (B, T, Nb, Dt), (B, N, C, F, T)
    taps = taps.float()
    band_bins = model.band_bins
    items = []  # (crop, stem, [refs], [queries])
    priors = []
    for b, c in enumerate(crops):
        emb = head_embeddings(taps[b], proj, head, pool)  # (Tp, Nb, e)
        frame_tok = torch.clamp(torch.arange(taps.shape[1], device=dev) // pool, max=emb.shape[0] - 1)
        priors.append(emb[frame_tok])
        refs = torch.from_numpy(c).to(dev)
        e_ref = refs.pow(2).flatten(1).sum(1)
        e_mix = mixes[b].pow(2).sum()
        present = (e_ref >= e_mix * 10 ** (present_db / 10)).cpu().numpy()
        if present.sum() < 2:
            continue
        P = _stft(refs, model.n_fft, model.hop).abs().pow(2)  # (J, C, F, T)
        Ps = spec[b].abs().pow(2)
        share = Ps / Ps.sum(0, keepdim=True).clamp_min(1e-20)
        stem_of = torch.einsum("jcft,ncft->jn", P, share).argmax(1).cpu().numpy()
        # token ownership among all references (the cache's energy, on this crop)
        pb = P.sum(1).double().cumsum(1)  # (J, F, T)
        pb = torch.cat([torch.zeros_like(pb[:, :1]), pb], 1)
        lo, hi = torch.as_tensor(band_bins[:, 0], device=dev), torch.as_tensor(band_bins[:, 1], device=dev)
        band = (pb[:, hi] - pb[:, lo]).float().transpose(1, 2)  # (J, T, Nb)
        tokE = torch.stack([_pool(x, pool, "sum") for x in band])  # (J, Tp, Nb)
        tot = tokE.sum(0)
        sh = tokE / tot.clamp_min(1e-20)
        aud = tot > tot.max() * 10 ** (-floor_db / 10)
        for s in range(spec.shape[1]):
            js = [j for j in np.where(present & (stem_of == s))[0]]
            if len(js) < 2:
                continue
            qs = []
            for j in js:
                ownd = aud & (sh[j] >= own)
                w = ownd.float() if int(ownd.sum()) >= 4 else sh[j] ** 2 * aud
                q = (emb * w[..., None]).sum((0, 1))
                if query_noise > 0:
                    q = unit(q) + query_noise * torch.randn_like(q) / math.sqrt(len(q))
                qs.append(unit(q))
            items.append((b, s, js, qs))
    if not items:
        return None
    order = rng.permutation(len(items))
    keep, n = [], 0
    for i in order:
        if n + len(items[i][2]) <= max_queries or not keep:
            keep.append(items[i])
            n += len(items[i][2])
    keep.sort(key=lambda it: (it[0], it[1]))
    item_crop = torch.tensor([b for b, _, _, _ in keep], device=dev)
    item_stem = torch.tensor([s for _, s, _, _ in keep], device=dev)
    stem_spec = torch.stack([spec[b, s] for b, s, _, _ in keep])
    feats = {b: stem_features(spec[b], band_bins) for b in {b for b, _, _, _ in keep}}
    stem_feat = torch.stack([feats[b][s] for b, s, _, _ in keep])
    q_item = torch.tensor([i for i, it in enumerate(keep) for _ in it[2]], device=dev)
    queries = torch.stack([q for it in keep for q in it[3]])
    refs = torch.stack([torch.from_numpy(crops[b][j]) for b, _, js, _ in keep for j in js]).to(dev)
    ref_power = _stft(refs, model.n_fft, model.hop).abs().pow(2)
    scale = torch.stack([mixes[b].pow(2).mean().sqrt() for b, _, js, _ in keep for _ in js]).clamp_min(1e-5)
    return Batch(taps, torch.stack(priors), item_crop, item_stem, stem_spec, stem_feat, queries, q_item, refs,
                 ref_power, scale, S)


# ------------------------------------------------------------------------------------------------ the loss

def _l1snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """(Q, ...) -> (Q,): 10 log10 of the L1 error over the reference's L1 (dB, lower is better)."""
    err = (est - ref).abs().flatten(1).mean(1)
    return 10 * torch.log10((err + eps) / (ref.abs().flatten(1).mean(1) + eps))


def decoder_loss(masks: torch.Tensor, batch: Batch, n_fft: int, hop: int, level_weight: float = 0.1,
                 resolutions: tuple[int, ...] = (512, 1024, 2048)) -> tuple[torch.Tensor, dict]:
    """Masks (Q, C, F, T) -> (loss, parts). Signals are divided by their crop's mix rms (loudness-free)."""
    est = _istft(batch.stem_spec[batch.q_item] * masks, n_fft, hop, batch.length)  # (Q, C, S)
    sc = batch.scale[:, None, None]
    est, ref = est / sc, batch.refs / sc
    wav = _l1snr(est, ref)
    spec = []
    for n in resolutions:
        if n > batch.length:
            continue
        E, R = _stft(est, n, n // 4).abs(), _stft(ref, n, n // 4).abs()
        spec.append(_l1snr(E, R))
    spec = torch.stack(spec).mean(0)
    ee, er = est.pow(2).flatten(1).mean(1), ref.pow(2).flatten(1).mean(1)
    level = F.relu(10 * torch.log10((er + 1e-8) / (ee + 1e-8)) - 3) / 10  # only when much quieter than the ref
    loss = (0.5 * (wav + spec) + level_weight * level).mean()
    return loss, {"wav_l1snr": float(wav.detach().mean()), "spec_l1snr": float(spec.detach().mean()),
                  "level": float(level.detach().mean())}


def _si_sdr_t(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r, e = ref.flatten(1).double(), est.flatten(1).double()
    a = (e * r).sum(1) / ((r * r).sum(1) + eps)
    return 10 * torch.log10((a * a * (r * r).sum(1) + eps) / ((a[:, None] * r - e).pow(2).sum(1) + eps))


@torch.no_grad()
def batch_scores(masks: torch.Tensor, batch: Batch, n_fft: int, hop: int) -> dict[str, np.ndarray]:
    """Per query: SI-SDR of the sub-source, of the whole SW stem (the baseline), of the ideal ratio mask
    within the stem's references (the bin-resolution ceiling), in dB."""
    stem = batch.stem_spec[batch.q_item]
    est = _istft(stem * masks, n_fft, hop, batch.length)
    sw = _istft(stem, n_fft, hop, batch.length)
    num = batch.ref_power
    den = num.new_zeros((len(batch.item_crop), *num.shape[1:])).index_add_(0, batch.q_item, num)[batch.q_item]
    irm = (num + 1e-12 / _counts(batch.q_item)[:, None, None, None]) / (den + 1e-12)  # sums to 1 per item
    ceil = _istft(stem * irm, n_fft, hop, batch.length)
    return {k: _si_sdr_t(batch.refs, v).cpu().numpy() for k, v in (("dec", est), ("sw", sw), ("irm", ceil))}


def _counts(q_item: torch.Tensor) -> torch.Tensor:
    return torch.bincount(q_item).float()[q_item]


# ---------------------------------------------------------------------------------------------- training

def _valid(decoder, model, head, proj, ds: SynthCrops, batch: int, pool: int, max_queries: int, **kw) -> dict:
    decoder.eval()
    rng = np.random.default_rng(0)
    dec, prior, sw, irm = [], [], [], []
    for i in range(0, len(ds), batch):
        b = prepare_batch(model, head, proj, [ds[j] for j in range(i, min(i + batch, len(ds)))], pool,
                          max_queries=10 ** 6, rng=rng, **kw)
        if b is None:
            continue
        with _autocast(model):
            m = b.decode(decoder)
        s = batch_scores(m.float(), b, model.n_fft, model.hop)
        dec.append(s["dec"]), sw.append(s["sw"]), irm.append(s["irm"])
    decoder.train()
    if not dec:
        return {"queries": 0}
    dec, sw, irm = np.concatenate(dec), np.concatenate(sw), np.concatenate(irm)
    return {"si_sdri": float(np.mean(dec - sw)), "si_sdri_median": float(np.median(dec - sw)),
            "si_sdr": float(np.mean(dec)), "si_sdri_irm": float(np.mean(irm - sw)), "queries": int(len(dec))}


def _autocast(model):
    dev = getattr(model, "device", "cpu")
    return torch.autocast("cuda", torch.bfloat16, enabled=str(dev).startswith("cuda"))


def train_decoder(model, head, proj, train_dirs: list, valid_dirs: list, out: str | Path, steps: int = 5000,
                  batch: int = 4, lr: float = 3e-4, crop_s: float = 6.0, max_queries: int = 24, dim: int = 128,
                  depth: int = 1, heads: int = 4, hidden: int = 256, pool: int = 4, own: float = 0.5,
                  present_db: float = -30.0, query_noise: float = 0.05, level_weight: float = 0.1,
                  valid_crops: int = 32, eval_every: int = 200, plateau: int = 5, plateau_delta: float = 0.05,
                  max_minutes: float = 120.0, seed: int = 0, workers: int = 2, log=_log,
                  extra: dict | None = None) -> dict:
    """Train a ``FineDecoder`` (see the module docstring). Every evaluation is appended to
    ``history.jsonl`` (and printed, flushed); ``best.pt`` keeps the best validation SI-SDR improvement over
    the SW stem; guards: non-finite loss, ``plateau`` evaluations without a gain of ``plateau_delta`` dB,
    ``max_minutes``. Step 0's validation is the prior alone (the registry's soft token split)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = getattr(model, "device", "cpu")
    train_songs, valid_songs = song_dirs(train_dirs), song_dirs(valid_dirs)
    if not train_songs or not valid_songs:
        raise ValueError("no songs found (folders of `priism-lab gen` songs, each with a meta.json)")
    sr = model.sample_rate
    tr = SynthCrops(train_songs, sr, crop_s, seed=seed)
    vr = np.random.default_rng(1234)
    va = SynthCrops(valid_songs, sr, crop_s)
    va.fixed = [va.pick(vr) for _ in range(valid_crops)]
    loader = torch.utils.data.DataLoader(tr, batch_size=batch, collate_fn=list, num_workers=workers,
                                         persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    kw = {"own": own, "present_db": present_db}
    tap_dim = None
    decoder = opt = sched = None
    hist = out / "history.jsonl"
    hist.unlink(missing_ok=True)

    def record(rec: dict) -> None:
        with hist.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log(json.dumps(rec))

    best, since, start, reason, step = -1e9, 0, time.time(), "steps", 0
    losses, parts_acc, skipped, t_step = [], [], 0, time.time()
    it = iter(loader)
    while step < steps:
        crops = next(it)
        b = prepare_batch(model, head, proj, crops, pool, max_queries=max_queries, query_noise=query_noise,
                          rng=rng, **kw)
        if b is None:
            skipped += 1
            continue
        if decoder is None:  # sizes known from the first batch
            tap_dim = b.taps.shape[-1]
            decoder = FineDecoder(tap_dim, model.band_bins, len(model.stems), b.queries.shape[-1],
                                  channels=b.refs.shape[1], dim=dim, depth=depth, heads=heads, hidden=hidden,
                                  pool=pool).to(dev)
            n_par = sum(p.numel() for p in decoder.parameters())
            opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
            v = _valid(decoder, model, head, proj, va, batch, pool, max_queries, **kw)
            record({"step": 0, "params": n_par, "valid": v, "note": "prior only (registry soft token split)"})
            best = v.get("si_sdri", -1e9)
            save_decoder(out / "best.pt", decoder, step=0, valid=v, **(extra or {}))
            t_step = time.time()
        step += 1
        with _autocast(model):
            masks = b.decode(decoder)
        loss, parts = decoder_loss(masks.float(), b, model.n_fft, model.hop, level_weight)
        if not torch.isfinite(loss):
            reason = f"non-finite loss at step {step}"
            log(reason)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(nn.utils.clip_grad_norm_(decoder.parameters(), 1.0))
        opt.step()
        sched.step()
        losses.append(loss.item())
        parts_acc.append({**parts, "grad_norm": gn, "queries": int(len(b.queries))})
        if step % eval_every == 0 or step == steps:
            sec = (time.time() - t_step) / len(losses)
            v = _valid(decoder, model, head, proj, va, batch, pool, max_queries, **kw)
            rec = {"step": step, "loss": float(np.mean(losses)),
                   **{k: float(np.mean([p[k] for p in parts_acc])) for k in parts_acc[0]},
                   "sec_per_step": sec, "skipped_batches": skipped, "minutes": (time.time() - start) / 60,
                   "prior_scale": float(decoder.prior_scale), "valid": v}
            if str(dev).startswith("cuda"):
                rec["max_vram_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            record(rec)
            losses, parts_acc = [], []
            if v.get("si_sdri", -1e9) > best + plateau_delta:
                best, since = v["si_sdri"], 0
                save_decoder(out / "best.pt", decoder, step=step, valid=v, **(extra or {}))
            else:
                since += 1
            if plateau and since >= plateau:
                reason = f"plateau: no gain of {plateau_delta} dB in {plateau} evaluations (step {step})"
                log(reason)
                break
            if rec["minutes"] > max_minutes:
                reason = f"time cap {max_minutes} min (step {step})"
                log(reason)
                break
            t_step = time.time()
    if decoder is not None:
        save_decoder(out / "last.pt", decoder, step=step, **(extra or {}))
    summary = {"stopped": reason, "best_si_sdri": best, "steps": step, "seed": seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------------------------- inference

@torch.no_grad()
def decoder_masks(model, decoder: FineDecoder, wav: np.ndarray, emb: np.ndarray, queries: list,
                  window_s: float = 8.0, margin_s: float = 1.0, log=None) -> list:
    """Whole song (C, S) at the model's rate, its unit head embeddings ``emb`` (Tp, Nb, e) on the
    analysis grid, and per stem the registry centroids (K, e), or None when the stem is not split ->
    per stem float16 masks (K, C, F, T) or None. A second SW pass in the analysis windows (the taps are not
    kept by ``analyse``: 1.6 GB for 5 minutes), the decoder on each window, the windows' centres kept."""
    from .separate import window_plan

    dev = next(decoder.parameters()).device
    S = wav.shape[1]
    n_frames = S // model.hop + 1
    win_f = int(round(window_s * model.sample_rate / model.hop))
    todo = [s for s, q in enumerate(queries) if q is not None and len(q) >= 2]
    out: list = [None] * len(queries)
    if not todo:
        return out
    emb_t = unit(torch.from_numpy(np.asarray(emb, np.float32)).to(dev))
    plan = window_plan(n_frames, model.hop, model.sample_rate, window_s, margin_s)
    q_all = torch.cat([unit(torch.as_tensor(np.asarray(queries[s], np.float32))) for s in todo]).to(dev)
    q_item = torch.tensor([i for i, s in enumerate(todo) for _ in range(len(queries[s]))], device=dev)
    pool = decoder.pool
    for w, (start_f, lo, hi) in enumerate(plan):
        seg = wav[:, start_f * model.hop: (start_f + win_f) * model.hop]
        if seg.shape[1] < win_f * model.hop:
            seg = np.pad(seg, ((0, 0), (0, win_f * model.hop - seg.shape[1])))
        taps, spec = model.taps_and_spec(torch.from_numpy(np.ascontiguousarray(seg, np.float32))[None].to(
            getattr(model, "device", "cpu")))
        taps, spec = taps.float().to(dev), spec.to(dev)
        T = taps.shape[1]
        tok = torch.clamp((start_f + torch.arange(T, device=dev)) // pool, max=emb_t.shape[0] - 1)
        feats = stem_features(spec[0], model.band_bins)
        idx = torch.as_tensor(todo, device=dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=str(dev).startswith("cuda")):
            m = decoder(taps, torch.zeros(len(todo), dtype=torch.long, device=dev), idx, spec[0, idx], feats[idx],
                        q_all, q_item, emb_t[tok][None])
        m = m.float()[..., lo:hi].cpu().numpy().astype(np.float16)
        g0, g1 = start_f + lo, start_f + hi
        for i, s in enumerate(todo):
            if out[s] is None:
                out[s] = np.zeros((len(queries[s]), *m.shape[1:3], n_frames), np.float16)
            out[s][..., g0:g1] = m[(q_item == i).cpu().numpy()]
        if log and (w + 1) % 20 == 0:
            log(f"  decoder window {w + 1}/{len(plan)}")
    return out


def render_masks(model, spec: np.ndarray, masks: np.ndarray, length: int):
    """Stem estimate (C, F, T) x K float16 masks (K, C, F, T) -> yields K waveforms (C, length). The last
    sub-source takes 1 - (sum of the others), so the sub-sources sum to the stem exactly despite float16."""
    acc = np.zeros(spec.shape, np.float32)
    K = len(masks)
    for k in range(K):
        m = masks[k].astype(np.float32) if k < K - 1 else 1 - acc
        if k < K - 1:
            acc += m
        yield model.istft(spec * m, length)


def oracle_bin_render(model, spec: np.ndarray, refs: np.ndarray, length: int):
    """Ideal ratio masks at bin resolution among one stem's references (J, C, S), applied to the stem
    estimate (C, F, T): the ceiling of any mask on the SW stem (the decoder's, at least without phase
    changes). Yields J waveforms; the masks sum to 1, so they sum to the stem. One reference spectrum at a
    time besides the sum (a 5-minute stereo power spectrum is 250 MB)."""
    def power(x):
        return _stft(torch.from_numpy(np.ascontiguousarray(x, np.float32)), model.n_fft, model.hop).abs().pow(2).numpy()

    J = len(refs)
    tot = np.zeros(spec.shape, np.float32)
    for r in refs:
        tot += power(r)[..., : spec.shape[-1]]
    for r in refs:
        m = (power(r)[..., : spec.shape[-1]] + 1e-12 / J) / (tot + 1e-12)
        yield model.istft(spec * m, length)
