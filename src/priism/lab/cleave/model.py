"""The Cleave JEPA (phase 1 of docs/cleave.md): a whole song in, one identity Z ~ N(mu, sigma^2) out.

Hierarchical, so a whole song fits: a local encoder reads each 2.2 s chunk (96 patches of 16 frames x 16
mel bands, plus a CLS token), a global encoder reads the CLS tokens of the visible chunks and a Z token.
The predictor sees only Z, the position of a patch and the deterministic P/L of its frames, never the
other patches: Z is a bottleneck that must carry what P and L do not explain, the timbre. Targets are the
patch tokens of an EMA copy of the local encoder on the unmasked chunk (I-JEPA).

Losses: MSE on the masked patches; a per-patch log-variance trained by Gaussian NLL on the detached error
(learning the variance never bends the prediction); beta * KL(Z || N(0, I)), the information bottleneck;
VICReg variance/covariance on the online patch tokens, because a predictor that only sees Z makes the
constant solution tempting.

Three uncertainty signals come out per song (docs/cleave.md 2.8): the posterior sigma of Z, the
predictor's variance, and its error.

V-bar (docs/cleave.md 2.9, ``v_dim`` > 0): a small latent per chunk, read from the CLS tokens of its
neighbours within ``v_window`` chunks (never its own chunk, which may be a target), given to the predictor
beside Z: the slot for mix effects, so they stop leaking into Z. With paired FX views of a crop, two more
losses push effects out of Z: invariance of the normalised mu between views, and swap prediction (the
targets of view A from Z of view B, V-bar and P/L of view A). ``v_dim`` = 0 is the phase 1a model, whose
checkpoints still load.

Context branch (``ctx_layers`` > 0, docs/cleave.md 3.2): the classic I-JEPA path beside the Z bottleneck. On
the band-masked chunks, a small predictor reads the online patch tokens of the chunk (the masked positions
hold the encoder's own mask-token outputs) and predicts the target tokens of the masked bands. Without it
the patch tokens get no prediction gradient (the main predictor reads only Z), so the EMA targets drifted to
high-rank noise and Z collapsed (probe s0b, 3.1). ``ctx_layers`` = 0 keeps the older models and checkpoints.

Mel targets (``target`` = "mel", docs/cleave-diagnostic.md): the prediction targets are the masked log-mel
patches themselves, normalised per patch (MAE's norm_pix_loss), instead of the EMA encoder's tokens. They
cannot drift or collapse, so the test of whether the Z bottleneck works no longer depends on the target
dynamics that broke every phase 1a probe.

Z-use hinge (``w_zuse`` > 0): z_use (the loss with Z of another song minus the loss with the true Z) becomes
a training term, max(0, ``zuse_margin`` - gap). Bounded on purpose: an open-ended gap is maximised by a
predictor that just breaks on a wrong Z, without Z learning anything. It is a contrastive term (close to
CPC / InfoNCE): Z learns whatever tells the songs of the batch apart, timbre or recording, so it has to be
measured, not assumed.

data2vec targets (``target`` = "d2v", Baevski et al. 2022, the usual recipe for audio): the mean of the EMA
encoder's last ``d2v_k`` layer outputs on the unmasked chunk, each instance-normalised (per channel over the
chunk's patches) before averaging. Several layers and a per-layer norm keep the targets from drifting to
one layer's noise or collapsing onto a few directions, without VICReg; the targets have unit variance per
channel, so beta keeps its intended scale.

Residual prediction (``residual``, docs/cleave-diagnostic.md section 8): a second predictor g reads only
position and P/L (its Z input is held at zero) and learns the targets on its own loss; the Z predictor learns
the residual t - g (g detached), standardised per dimension over the batch. Everything P/L explain is gone
from what Z must predict, and the residual has unit variance, so Z's gain is no longer ~0.003 of MSE that
any KL outbids. ``pred`` is then the MSE on the standardised residual: ~1 means Z adds nothing.

V-bar read from the stem's own neighbours carries its timbre, a free channel (no KL) that let the
predictor stop using Z. Three counters, each optional: the swap reads V-bar from another stem of the same
song under chain A (``pair`` with an other view), so V-bar only helps there by carrying the chain;
``w_vinv`` asks V-bar to be the same from the stem and from that other stem; ``v_drop`` zeroes V-bar per
item in the main prediction, so Z alone must still predict.

The invariance of Z between views is InfoNCE by default (``inv="nce"``): mu_B must be closer to its own
mu_A than to the other crops of the batch. The plain distance between normalised mu (``inv="mse"``) is
minimised by a constant mu, and in the 2.9 smoke run Z collapsed to one (std across crops 0.0015, against
0.28 without pairs; the effective rank, computed on normalised mu, did not show it).
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .features import N_MELS, PL_DIM


@dataclass
class CleaveConfig:
    dim: int = 256
    heads: int = 4
    local_layers: int = 4
    global_layers: int = 4
    pred_layers: int = 4
    z_dim: int = 64
    chunk_frames: int = 192
    patch_t: int = 16
    patch_f: int = 16
    n_mels: int = N_MELS
    pl_dim: int = PL_DIM
    v_dim: int = 0  # 0: no V-bar (phase 1a checkpoints)
    v_window: int = 2  # V-bar of chunk c reads chunks c-w..c+w except c
    ctx_layers: int = 0  # 0: no context branch (checkpoints before 3.2)
    target: str = "ema"  # "ema": EMA encoder tokens (I-JEPA); "mel": log-mel patches (MAE); "d2v": data2vec
    d2v_k: int = 2  # d2v: layers averaged (the last ones)
    residual: bool = False  # Z predicts what a P/L-only predictor leaves (standardised)
    p_mode: str = "salience"  # P factor, features.P_MODES ("salience" for checkpoints before 9.2)
    l_mode: str = "frame"  # loudness channel, features.L_MODES
    onset_mode: str = "flux"  # onset channel, features.ONSET_MODES

    @property
    def n_tcol(self) -> int:
        return self.chunk_frames // self.patch_t

    @property
    def n_rows(self) -> int:
        return self.n_mels // self.patch_f

    @property
    def n_patches(self) -> int:
        return self.n_tcol * self.n_rows

    @property
    def target_dim(self) -> int:
        return self.patch_t * self.patch_f if self.target == "mel" else self.dim

    def to_dict(self) -> dict:
        return asdict(self)


def _body(dim: int, heads: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def patchify(chunks: torch.Tensor, cfg: CleaveConfig) -> torch.Tensor:
    """(N, chunk_frames, n_mels) -> (N, n_patches, patch_t * patch_f), patch index = tcol * n_rows + row."""
    n = chunks.shape[0]
    x = chunks.reshape(n, cfg.n_tcol, cfg.patch_t, cfg.n_rows, cfg.patch_f)
    return x.permute(0, 1, 3, 2, 4).reshape(n, cfg.n_patches, cfg.patch_t * cfg.patch_f)


def sinusoid(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """(C,) positions -> (C, dim): any song length, no learned table to outgrow."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=pos.device) / half)
    ang = pos.float()[:, None] * freq[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class LocalEncoder(nn.Module):
    def __init__(self, cfg: CleaveConfig):
        super().__init__()
        d = cfg.dim
        self.embed = nn.Linear(cfg.patch_t * cfg.patch_f, d)
        self.pos = nn.Parameter(torch.randn(1, cfg.n_patches, d) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.body = _body(d, cfg.heads, cfg.local_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, patches: torch.Tensor, mask: torch.Tensor | None = None):
        """(N, P, in), (N, P) bool -> CLS (N, d), patch tokens (N, P, d)."""
        x = self.embed(patches)
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype).expand_as(x), x)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x + self.pos], 1)
        x = self.norm(self.body(x))
        return x[:, 0], x[:, 1:]

    def layer_targets(self, patches: torch.Tensor, k: int) -> torch.Tensor:
        """data2vec targets: (N, P, in) -> (N, P, d), mean of the last ``k`` layer outputs (patch tokens),
        each instance-normalised over the P patches of its chunk."""
        x = torch.cat([self.cls.expand(patches.shape[0], -1, -1), self.embed(patches) + self.pos], 1)
        outs = []
        for layer in self.body.layers:
            x = layer(x)
            outs.append(x[:, 1:])
        tops = [F.instance_norm(o.float().transpose(1, 2)).transpose(1, 2) for o in outs[-k:]]
        return torch.stack(tops).mean(0)


class GlobalEncoder(nn.Module):
    def __init__(self, cfg: CleaveConfig):
        super().__init__()
        d = cfg.dim
        self.z_token = nn.Parameter(torch.zeros(1, 1, d))
        self.body = _body(d, cfg.heads, cfg.global_layers)
        self.norm = nn.LayerNorm(d)
        self.mu = nn.Linear(d, cfg.z_dim)
        self.logvar = nn.Linear(d, cfg.z_dim)

    def forward(self, cls: torch.Tensor, pad: torch.Tensor):
        """(B, C, d) chunk tokens, (B, C) True = not read -> mu, logvar (B, z_dim)."""
        B, C, d = cls.shape
        x = cls + sinusoid(torch.arange(C, device=cls.device), d).to(cls.dtype)
        x = torch.cat([self.z_token.expand(B, -1, -1).to(x.dtype), x], 1)
        pad = torch.cat([pad.new_zeros(B, 1), pad], 1)
        h = self.norm(self.body(x, src_key_padding_mask=pad))[:, 0]
        return self.mu(h), self.logvar(h).clamp(-10, 10)


class Predictor(nn.Module):
    def __init__(self, cfg: CleaveConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.pl_embed = nn.Linear(cfg.patch_t * cfg.pl_dim, d)
        self.z_embed = nn.Linear(cfg.z_dim, d)
        self.v_embed = nn.Linear(cfg.v_dim, d) if cfg.v_dim else None
        self.pos = nn.Parameter(torch.randn(1, cfg.n_patches, d) * 0.02)
        self.body = _body(d, cfg.heads, cfg.pred_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.target_dim)
        self.logv = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor, pl: torch.Tensor, v: torch.Tensor | None = None):
        """(N, z_dim), (N, n_tcol, patch_t * pl_dim), (N, v_dim) -> predicted tokens (N, P, d), log-var (N, P)."""
        t = self.pl_embed(pl).repeat_interleave(self.cfg.n_rows, dim=1)  # one P/L per time column
        x = t + self.pos + self.z_embed(z)[:, None]
        if self.v_embed is not None:
            x = x + self.v_embed(v)[:, None]
        h = self.norm(self.body(x))
        return self.out(h), self.logv(h).squeeze(-1).clamp(-10, 10)


class VBar(nn.Module):
    """Chunk CLS tokens -> V-bar per chunk: the mean of the encoded neighbours' CLS within ``v_window``
    (own chunk excluded), through an MLP to ``v_dim``. A chunk with no encoded neighbour gets zeros."""

    def __init__(self, cfg: CleaveConfig):
        super().__init__()
        d = cfg.dim
        self.w = cfg.v_window
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.v_dim))

    def forward(self, cls: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        """(B, C, d), (B, C) bool encoded -> (B, C, v_dim)."""
        i = torch.arange(cls.shape[1], device=cls.device)
        gap = (i[:, None] - i[None]).abs()
        band = ((gap <= self.w) & (gap > 0)).to(cls.dtype)
        e = enc.to(cls.dtype)
        num = torch.einsum("ij,bjd->bid", band, cls * e[..., None])
        den = torch.einsum("ij,bj->bi", band, e)
        h = num / den.clamp_min(1)[..., None]
        return self.mlp(self.norm(h)) * (den > 0)[..., None].to(cls.dtype)


class ContextPredictor(nn.Module):
    """Online patch tokens of one chunk -> predicted target tokens at every position (I-JEPA)."""

    def __init__(self, cfg: CleaveConfig):
        super().__init__()
        d = cfg.dim
        self.pos = nn.Parameter(torch.randn(1, cfg.n_patches, d) * 0.02)
        self.body = _body(d, cfg.heads, cfg.ctx_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.target_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.body(tokens + self.pos)))


def vicreg(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, d) -> variance hinge (std >= 1 per dim) and off-diagonal covariance (Bardes et al. 2022)."""
    x = x.float() - x.float().mean(0)
    std = torch.sqrt(x.var(0) + 1e-4)
    cov = x.T @ x / max(1, x.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return torch.relu(1 - std).mean(), (off**2).sum() / x.shape[1]


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) of centred rows (Roy & Vetterli 2007)."""
    x = x.float() - x.float().mean(0)
    s = torch.linalg.svdvals(x)
    p = s / (s.sum() + 1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


class Cleave(nn.Module):
    def __init__(self, cfg: CleaveConfig):
        super().__init__()
        self.cfg = cfg
        self.local = LocalEncoder(cfg)
        self.glob = GlobalEncoder(cfg)
        self.pred = Predictor(cfg)
        self.vbar = VBar(cfg) if cfg.v_dim else None
        self.ctx = ContextPredictor(cfg) if cfg.ctx_layers else None
        self.base = Predictor(cfg) if cfg.residual else None  # g(P, L): fed a zero Z
        self.target = copy.deepcopy(self.local)
        self.target.requires_grad_(False)

    @torch.no_grad()
    def ema(self, m: float) -> None:
        for t, o in zip(self.target.parameters(), self.local.parameters()):
            t.mul_(m).add_(o.detach(), alpha=1 - m)

    def chunk(self, logmel: torch.Tensor, pl: torch.Tensor):
        """(B, T, n_mels), (B, T, pl_dim) with T >= C * chunk_frames -> patches (B, C, P, in), pl (B, C, n_tcol, .)."""
        cfg = self.cfg
        B = logmel.shape[0]
        C = logmel.shape[1] // cfg.chunk_frames
        T = C * cfg.chunk_frames
        patches = patchify(logmel[:, :T].reshape(B * C, cfg.chunk_frames, -1), cfg).view(B, C, cfg.n_patches, -1)
        pl = pl[:, :T].reshape(B, C, cfg.n_tcol, cfg.patch_t * cfg.pl_dim)
        return patches, pl

    def encode(self, patches: torch.Tensor, active: torch.Tensor, pmask: torch.Tensor | None = None,
               full: bool = False):
        """Posterior of Z from the visible patches: mu, logvar (B, z_dim), the online patch tokens read, and
        V-bar (B, C, v_dim), None without V-bar. ``full`` adds (batch idx, chunk idx, all patch tokens) of
        the encoded chunks, for the context branch."""
        B, C, P, _ = patches.shape
        if pmask is None:
            pmask = torch.zeros(B, C, P, dtype=torch.bool, device=patches.device)
        enc = active & ~pmask.all(-1)
        enc[~enc.any(1), 0] = True  # an all-silent crop still gives the global encoder one token
        bi, ci = enc.nonzero(as_tuple=True)
        cls_e, tok_e = self.local(patches[bi, ci], pmask[bi, ci])
        cls = cls_e.new_zeros(B, C, cls_e.shape[-1])
        cls = cls.index_put((bi, ci), cls_e)
        mu, logvar = self.glob(cls, ~enc)
        v = self.vbar(cls, enc) if self.vbar is not None else None
        if full:
            return mu, logvar, tok_e[~pmask[bi, ci]], v, (bi, ci, tok_e)
        return mu, logvar, tok_e[~pmask[bi, ci]], v

    def forward(self, patches: torch.Tensor, pl: torch.Tensor, active: torch.Tensor, pmask: torch.Tensor,
                beta: float = 1e-3, w_var: float = 1.0, w_cov: float = 0.04, shuffle_z: bool = False,
                pair: tuple | None = None, w_inv: float = 1.0, w_swap: float = 1.0, v_drop: float = 0.0,
                w_vinv: float = 0.0, inv_mode: str = "nce", inv_tau: float = 0.1, w_ctx: float = 1.0, w_zuse: float = 0.0,
                zuse_margin: float = 0.1) -> dict:
        """``pair`` = (batch positions of the paired items, view B patches, view B active chunks), optionally
        followed by (other-stem patches, other-stem active chunks): view A is the item itself. Views B and O
        are encoded under view A's mask, so Z_B never reads the sections A must predict and V-bar_O sees the
        same neighbours as V-bar_A."""
        B = patches.shape[0]
        mu, logvar, tokens, v, (bi, ci, tok_all) = self.encode(patches, active, pmask, full=True)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        if shuffle_z:  # Z of another song: how much the predictor relies on Z
            z = z.roll(1, 0)
        tgt = active & pmask.any(-1)
        tb, tc = tgt.nonzero(as_tuple=True)
        with torch.no_grad():
            if self.cfg.target == "mel":
                t = F.layer_norm(patches[tb, tc].float(), (patches.shape[-1],))
            elif self.cfg.target == "d2v":  # already normalised per channel, layer by layer
                t = self.target.layer_targets(patches[tb, tc], self.cfg.d2v_k)
            else:
                _, t = self.target(patches[tb, tc])
                t = F.layer_norm(t.float(), (t.shape[-1],))
        vt = v[tb, tc] if v is not None else None
        if vt is not None and self.training and v_drop > 0:  # Z alone must predict these items
            vt = vt * (torch.rand(B, device=vt.device) >= v_drop).to(vt.dtype)[tb, None]
        m = pmask[tb, tc]
        loss_base = torch.zeros((), device=patches.device)
        if self.base is not None:
            zero = torch.zeros_like(z[tb])
            g, _ = self.base(zero, pl[tb, tc], torch.zeros_like(vt) if vt is not None else None)
            loss_base = ((g.float() - t) ** 2).mean(-1)[m].mean()
            r = t - g.detach().float()
            t = r / r[m].std(0, keepdim=True).clamp_min(1e-4)  # the Z predictor's target from here on
        pred, logv = self.pred(z[tb], pl[tb, tc], vt)
        err = ((pred.float() - t) ** 2).mean(-1)
        logv = logv.float()
        loss_pred = err[m].mean()
        loss_nll = (0.5 * (logv + err.detach() / logv.exp()))[m].mean()
        kl = (0.5 * (mu.float() ** 2 + logvar.float().exp() - 1 - logvar.float())).sum(-1).mean()
        rows = tokens if tokens.shape[0] <= 4096 else tokens[torch.randperm(tokens.shape[0], device=tokens.device)[:4096]]
        v_loss, c_loss = vicreg(rows)
        loss = loss_pred + loss_nll + beta * kl + w_var * v_loss + w_cov * c_loss + loss_base
        zgap = torch.zeros((), device=loss.device)
        if w_zuse > 0 and B > 1:  # the Z of the next crop of the batch: another song (or another stem)
            ps, _ = self.pred(z.roll(1, 0)[tb], pl[tb, tc], vt)
            zgap = ((ps.float() - t) ** 2).mean(-1)[m].mean() - loss_pred
            loss = loss + w_zuse * torch.relu(zuse_margin - zgap)
        ctx = torch.zeros((), device=loss.device)
        if self.ctx is not None:  # band-masked chunks: encoded, and target rows of their own
            row = torch.full(pmask.shape[:2], -1, dtype=torch.long, device=pmask.device)
            row[bi, ci] = torch.arange(len(bi), device=pmask.device)
            r = row[tb, tc]
            sel = r >= 0
            if sel.any():
                pc = self.ctx(tok_all[r[sel]])
                ctx = ((pc.float() - t[sel]) ** 2).mean(-1)[m[sel]].mean()
                loss = loss + w_ctx * ctx
        inv = swap = vinv = torch.zeros((), device=loss.device)
        if pair is not None and len(pair[0]) > 0:
            pi, patches_b, active_b = pair[:3]
            mu_b, logvar_b, _, _ = self.encode(patches_b, active_b, pmask[pi])
            v_o = self.encode(pair[3], pair[4], pmask[pi])[3] if len(pair) > 3 and v is not None else None
            if v_o is not None and w_vinv > 0:  # V-bar from the stem and from another stem, same chain
                both = (v[pi].abs().sum(-1) > 0) & (v_o.abs().sum(-1) > 0)
                if both.any():
                    d = F.normalize(v[pi].float(), dim=-1) - F.normalize(v_o.float(), dim=-1)
                    vinv = (d ** 2).sum(-1)[both].mean()
            z_b = mu_b + torch.exp(0.5 * logvar_b) * torch.randn_like(mu_b) if self.training else mu_b
            na, nb = F.normalize(mu.float(), dim=-1), F.normalize(mu_b.float(), dim=-1)
            if inv_mode == "nce":  # the other crops of the batch are the negatives
                inv = F.cross_entropy(nb @ na.T / inv_tau, pi)
            else:
                inv = ((na[pi] - nb) ** 2).sum(-1).mean()
            pos = torch.full((B,), -1, dtype=torch.long, device=patches.device)
            pos[pi] = torch.arange(len(pi), device=patches.device)
            sel = pos[tb] >= 0
            if sel.any():
                if v_o is not None:
                    vs = v_o[pos[tb[sel]], tc[sel]]
                else:
                    vs = v[tb[sel], tc[sel]] if v is not None else None
                ps, _ = self.pred(z_b[pos[tb[sel]]], pl[tb[sel], tc[sel]], vs)
                swap = ((ps.float() - t[sel]) ** 2).mean(-1)[m[sel]].mean()
            kl_b = (0.5 * (mu_b.float() ** 2 + logvar_b.float().exp() - 1 - logvar_b.float())).sum(-1).mean()
            loss = loss + w_inv * inv + w_swap * swap + w_vinv * vinv + beta * kl_b * len(pi) / B

        # per-song signals (docs/cleave.md 2.8)
        mf = m.float()
        n_t = torch.zeros(B, device=mf.device).index_add_(0, tb, mf.sum(1)).clamp_min(1)
        err_song = torch.zeros(B, device=mf.device).index_add_(0, tb, (err.detach() * mf).sum(1)) / n_t
        var_song = torch.zeros(B, device=mf.device).index_add_(0, tb, (logv.detach().exp() * mf).sum(1)) / n_t
        sigma_song = torch.exp(0.5 * logvar.detach().float()).mean(-1)
        return {"loss": loss, "pred": loss_pred.detach(), "nll": loss_nll.detach(), "kl": kl.detach(),
                "vic_var": v_loss.detach(), "vic_cov": c_loss.detach(), "ctx": ctx.detach(), "zgap": zgap.detach(), "base": loss_base.detach(), "inv": inv.detach(),
                "swap": swap.detach(), "vinv": vinv.detach(), "mu": mu.detach(),
                "sigma": sigma_song, "pred_var": var_song, "pred_err": err_song, "targets": t}
