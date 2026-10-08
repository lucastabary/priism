"""The attractor separator: band-split core, attractor decoder, FiLM mask head."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn


@dataclass
class SeparatorConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    hop: int = 512
    dim: int = 128
    depth: int = 4  # core layers, each = one time-attention + one band-attention block
    heads: int = 4
    max_sources: int = 16  # K: number of attractor slots
    decoder_depth: int = 2
    head_hidden: int = 256
    # Bins per band, low to high; must sum to n_fft // 2 + 1. Empty = a default split.
    bands: list[int] = field(default_factory=list)

    def band_sizes(self) -> list[int]:
        if self.bands:
            assert sum(self.bands) == self.n_fft // 2 + 1, "bands must cover every STFT bin"
            return list(self.bands)
        return default_bands(self.n_fft // 2 + 1)

    def to_dict(self) -> dict:
        return asdict(self)


def default_bands(n_bins: int) -> list[int]:
    """Narrow bands in the low end, wider ones up high (like BS-RoFormer's split)."""
    sizes, total, width = [], 0, 2
    while total < n_bins:
        for _ in range(8):
            if total >= n_bins:
                break
            w = min(width, n_bins - total)
            sizes.append(w)
            total += w
        width *= 2
    return sizes


def _sinusoid(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    ang = pos / (10000 ** (i / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class BandSplit(nn.Module):
    """Each band's complex stereo bins -> one D-dim vector per frame."""

    def __init__(self, bands: list[int], dim: int):
        super().__init__()
        self.bands = bands
        self.proj = nn.ModuleList(nn.Sequential(nn.LayerNorm(4 * b), nn.Linear(4 * b, dim)) for b in bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F, 4)
        out, f = [], 0
        for b, proj in zip(self.bands, self.proj):
            out.append(proj(x[:, :, f:f + b].flatten(2)))
            f += b
        return torch.stack(out, dim=2)  # (B, T, Nb, D)


def _block(dim: int, heads: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True,
                                      activation="gelu")


class Core(nn.Module):
    """Alternating attention along time (per band) and across bands (per frame)."""

    def __init__(self, n_bands: int, dim: int, depth: int, heads: int):
        super().__init__()
        self.band_emb = nn.Parameter(torch.zeros(n_bands, dim))
        self.time_blocks = nn.ModuleList(_block(dim, heads) for _ in range(depth))
        self.band_blocks = nn.ModuleList(_block(dim, heads) for _ in range(depth))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, T, Nb, D)
        B, T, Nb, D = h.shape
        h = h + self.band_emb + _sinusoid(T, D, h.device)[None, :, None]
        for tb, bb in zip(self.time_blocks, self.band_blocks):
            h = tb(h.permute(0, 2, 1, 3).reshape(B * Nb, T, D)).reshape(B, Nb, T, D).permute(0, 2, 1, 3)
            h = bb(h.reshape(B * T, Nb, D)).reshape(B, T, Nb, D)
        return h


class AttractorDecoder(nn.Module):
    """K learned slot queries read the mix and become identity vectors with an existence logit."""

    def __init__(self, dim: int, heads: int, depth: int, k: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, depth)
        self.exist = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, extra: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, Nb, D = h.shape
        # Memory: one token per (frame, band) would be T*Nb long; per-frame and per-band summaries keep it short.
        memory = torch.cat([h.mean(2), h.mean(1)], dim=1)  # (B, T + Nb, D)
        if extra is not None:  # finer tokens (e.g. a time-frequency grid) next to the summaries
            memory = torch.cat([memory, extra], dim=1)
        a = self.decoder(self.queries.expand(B, -1, -1), memory)  # (B, K, D)
        return a, self.exist(a).squeeze(-1)


class MaskHead(nn.Module):
    """Shared per-band MLP, conditioned on one attractor by FiLM, giving a complex stereo mask."""

    def __init__(self, bands: list[int], dim: int, hidden: int):
        super().__init__()
        self.bands = bands
        self.film = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim)
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 2 * 4 * b))
                                  for b in bands)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # h: (B, T, Nb, D), a: (B, K, D) -> mask (B, K, T, F, 2 ch, 2 re/im)
        gamma, beta = self.film(a).chunk(2, dim=-1)
        x = self.norm(h)[:, None] * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]  # (B, K, T, Nb, D)
        out = []
        for i, (b, mlp) in enumerate(zip(self.bands, self.mlps)):
            y = mlp(x[:, :, :, i])  # (B, K, T, 8b)
            y = nn.functional.glu(y, dim=-1)  # (B, K, T, 4b)
            out.append(y.unflatten(-1, (b, 2, 2)))
        return torch.cat(out, dim=3)


class AttractorSeparator(nn.Module):
    def __init__(self, cfg: SeparatorConfig):
        super().__init__()
        self.cfg = cfg
        bands = cfg.band_sizes()
        self.split = BandSplit(bands, cfg.dim)
        self.core = Core(len(bands), cfg.dim, cfg.depth, cfg.heads)
        self.attractors = AttractorDecoder(cfg.dim, cfg.heads, cfg.decoder_depth, cfg.max_sources)
        self.head = MaskHead(bands, cfg.dim, cfg.head_hidden)
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, 2, S) -> complex (B, 2, F, T)."""
        B, C, S = wav.shape
        spec = torch.stft(wav.reshape(B * C, S), self.cfg.n_fft, self.cfg.hop, window=self.window, return_complex=True)
        return spec.reshape(B, C, *spec.shape[-2:])

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        lead = spec.shape[:-2]
        wav = torch.istft(spec.reshape(-1, *spec.shape[-2:]), self.cfg.n_fft, self.cfg.hop, window=self.window,
                          length=length)
        return wav.reshape(*lead, length)

    def forward(self, mix: torch.Tensor) -> dict[str, torch.Tensor]:
        """mix (B, 2, S) -> {"sources": (B, K, 2, S), "exist_logits": (B, K), "attractors": (B, K, D)}."""
        S = mix.shape[-1]
        spec = self.stft(mix)  # (B, 2, F, T)
        x = torch.view_as_real(spec).permute(0, 3, 2, 1, 4).flatten(3)  # (B, T, F, 4)
        h = self.core(self.split(x))
        a, exist = self.attractors(h)
        m = self.head(h, a)  # (B, K, T, F, 2, 2)
        mask = torch.view_as_complex(m.contiguous()).permute(0, 1, 4, 3, 2)  # (B, K, 2, F, T)
        est = self.istft(mask * spec[:, None], S)
        return {"sources": est, "exist_logits": exist, "attractors": a}

    @torch.no_grad()
    def separate(self, mix: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """One mix (2, S) -> (kept sources (N, 2, S), rest (2, S)); rest makes the sum exact."""
        out = self.forward(mix[None])
        keep = torch.sigmoid(out["exist_logits"][0]) > threshold
        src = out["sources"][0][keep]
        return src, mix - src.sum(0)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

