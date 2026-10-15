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

    @property
    def n_tcol(self) -> int:
        return self.chunk_frames // self.patch_t

    @property
    def n_rows(self) -> int:
        return self.n_mels // self.patch_f

    @property
    def n_patches(self) -> int:
        return self.n_tcol * self.n_rows

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
        self.pos = nn.Parameter(torch.randn(1, cfg.n_patches, d) * 0.02)
        self.body = _body(d, cfg.heads, cfg.pred_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        self.logv = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor, pl: torch.Tensor):
        """(N, z_dim), (N, n_tcol, patch_t * pl_dim) -> predicted tokens (N, P, d), log-variance (N, P)."""
        t = self.pl_embed(pl).repeat_interleave(self.cfg.n_rows, dim=1)  # one P/L per time column
        x = t + self.pos + self.z_embed(z)[:, None]
        h = self.norm(self.body(x))
        return self.out(h), self.logv(h).squeeze(-1).clamp(-10, 10)


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

    def encode(self, patches: torch.Tensor, active: torch.Tensor, pmask: torch.Tensor | None = None):
        """Posterior of Z from the visible patches: mu, logvar (B, z_dim), and the online patch tokens read."""
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
        return mu, logvar, tok_e[~pmask[bi, ci]]

    def forward(self, patches: torch.Tensor, pl: torch.Tensor, active: torch.Tensor, pmask: torch.Tensor,
                beta: float = 1e-3, w_var: float = 1.0, w_cov: float = 0.04, shuffle_z: bool = False) -> dict:
        B = patches.shape[0]
        mu, logvar, tokens = self.encode(patches, active, pmask)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        if shuffle_z:  # Z of another song: how much the predictor relies on Z
            z = z.roll(1, 0)
        tgt = active & pmask.any(-1)
        tb, tc = tgt.nonzero(as_tuple=True)
        with torch.no_grad():
            _, t = self.target(patches[tb, tc])
            t = F.layer_norm(t.float(), (t.shape[-1],))
        pred, logv = self.pred(z[tb], pl[tb, tc])
        m = pmask[tb, tc]
        err = ((pred.float() - t) ** 2).mean(-1)
        logv = logv.float()
        loss_pred = err[m].mean()
        loss_nll = (0.5 * (logv + err.detach() / logv.exp()))[m].mean()
        kl = (0.5 * (mu.float() ** 2 + logvar.float().exp() - 1 - logvar.float())).sum(-1).mean()
        rows = tokens if tokens.shape[0] <= 4096 else tokens[torch.randperm(tokens.shape[0], device=tokens.device)[:4096]]
        v_loss, c_loss = vicreg(rows)
        loss = loss_pred + loss_nll + beta * kl + w_var * v_loss + w_cov * c_loss

        # per-song signals (docs/cleave.md 2.8)
        mf = m.float()
        n_t = torch.zeros(B, device=mf.device).index_add_(0, tb, mf.sum(1)).clamp_min(1)
        err_song = torch.zeros(B, device=mf.device).index_add_(0, tb, (err.detach() * mf).sum(1)) / n_t
        var_song = torch.zeros(B, device=mf.device).index_add_(0, tb, (logv.detach().exp() * mf).sum(1)) / n_t
        sigma_song = torch.exp(0.5 * logvar.detach().float()).mean(-1)
        return {"loss": loss, "pred": loss_pred.detach(), "nll": loss_nll.detach(), "kl": kl.detach(),
                "vic_var": v_loss.detach(), "vic_cov": c_loss.detach(), "mu": mu.detach(),
                "sigma": sigma_song, "pred_var": var_song, "pred_err": err_song, "targets": t}
