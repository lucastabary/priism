"""Attractor separator on top of a pretrained MSST BS-RoFormer (e.g. BS-Roformer-SW).

The pretrained band-split core keeps everything it knows about music. Its
fixed per-stem mask estimators are replaced by one estimator, initialised from
a pretrained one (``other``, index 2 in BS-Roformer-SW) and conditioned by FiLM on each
attractor. FiLM starts near identity, so at step 0 every slot behaves like the
pretrained head and training only has to teach slots to specialise.

Needs the MSST repository on ``sys.path`` (``msst_path``), as on the pod.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .separator import AttractorDecoder


def load_msst_roformer(config_path: str | Path, ckpt_path: str | Path | None, msst_path: str | Path) -> nn.Module:
    """Build MSST's BSRoformer from its YAML config and load the checkpoint (weights only)."""
    import yaml

    if str(msst_path) not in sys.path:
        sys.path.insert(0, str(msst_path))
    from models.bs_roformer.bs_roformer import BSRoformer  # type: ignore

    cfg = yaml.load(Path(config_path).read_text(), Loader=yaml.FullLoader)
    model = BSRoformer(**dict(cfg["model"]))
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
        model.load_state_dict(state)
    return model


def _sinusoid(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-9.2 / dim))
    return torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], dim=-1)


class GridMemory(nn.Module):
    """Time-frequency grid for the attractors to read (v2): frames pooled by ``pool``, every band kept, each
    token marked with its band and time. The v1 summaries (band mean per frame, time mean per band) lose
    which notes play when, the only thing that tells two parts of the same instrument apart."""

    def __init__(self, dim: int, n_bands: int, pool: int = 4):
        super().__init__()
        self.pool = pool
        self.band = nn.Parameter(torch.zeros(n_bands, dim))
        self.time = nn.Linear(dim, dim)
        nn.init.zeros_(self.time.weight)
        nn.init.zeros_(self.time.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, T, Nb, D) -> (B, T/pool * Nb, D)
        B, T, Nb, D = h.shape
        g = nn.functional.avg_pool1d(h.permute(0, 2, 3, 1).reshape(B, Nb * D, T), self.pool, ceil_mode=True)
        g = g.reshape(B, Nb, D, -1).permute(0, 3, 1, 2)  # (B, T', Nb, D)
        g = g + self.band + self.time(_sinusoid(g.shape[1], D, h.device).to(g.dtype))[:, None]
        return g.flatten(1, 2)


class SlotRefiner(nn.Module):
    """Per-slot context after FiLM (v2): dilated convolutions along time and attention across bands, in a
    small width, added back through a zero-initialised projection (identity at step 0). The pretrained mask
    head sees one frame and one band at a time; this lets a slot follow its own line through time."""

    def __init__(self, dim: int, width: int = 64, heads: int = 4, dilations: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.convs = nn.ModuleList(nn.Conv1d(width, 2 * width, 5, padding=2 * d, dilation=d, groups=width)
                                   for d in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in dilations)
        self.freq = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _time(y, conv, norm):
        N, T, Nb, W = y.shape
        z = norm(y).permute(0, 2, 3, 1).reshape(N * Nb, W, T)
        return y + nn.functional.glu(conv(z), dim=1).reshape(N, Nb, W, T).permute(0, 3, 1, 2)

    def _bands(self, y):
        N, T, Nb, W = y.shape
        return self.freq(y.reshape(N * T, Nb, W) + _sinusoid(Nb, W, y.device).to(y.dtype)).reshape(N, T, Nb, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, T, Nb, D)
        # One checkpoint per block: B*K slots at full resolution would not fit in 24 GB otherwise.
        ck = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if self.training and torch.is_grad_enabled() \
            else (lambda f, *a: f(*a))
        y = self.inp(x)
        for conv, norm in zip(self.convs, self.norms):
            y = ck(self._time, y, conv, norm)
        return x + self.out(ck(self._bands, y))


class MsstAttractorSeparator(nn.Module):
    def __init__(self, roformer: nn.Module, max_sources: int = 16, decoder_depth: int = 2, heads: int = 8,
                 init_head: int = 2, film_init_std: float = 0.01, grad_checkpoint: bool = False, v2: bool = False):
        super().__init__()
        self.r = roformer
        self.grad_checkpoint = grad_checkpoint  # recompute activations in backward: long chunks fit in 24 GB
        dim = roformer.final_norm.gamma.shape[-1]
        self.attractors = AttractorDecoder(dim, heads, decoder_depth, max_sources)
        self.film = nn.Linear(dim, 2 * dim)
        nn.init.normal_(self.film.weight, std=film_init_std)
        nn.init.zeros_(self.film.bias)
        self.head = copy.deepcopy(roformer.mask_estimators[init_head])
        del self.r.mask_estimators  # the fixed stems are gone; only the shared conditioned head remains
        self.v2 = v2
        if v2:  # twins: attractors read a time-frequency grid, slots get their own context (see the classes)
            self.grid = GridMemory(dim, len(roformer.band_split.to_features))
            self.refiner = SlotRefiner(dim)

    def _features(self, raw: torch.Tensor):
        """Same steps as BSRoformer.forward up to final_norm. raw (B, C, S) -> x (B, T, Nb, D), stft (B, F*C, T, 2)."""
        from einops import pack, rearrange, unpack

        r = self.r
        B, C, S = raw.shape
        window = r.stft_window_fn(device=raw.device)
        spec = torch.stft(raw.reshape(B * C, S), **r.stft_kwargs, window=window, return_complex=True)
        spec = torch.view_as_real(spec).reshape(B, C, *spec.shape[-2:], 2)
        stft_repr = rearrange(spec, "b s f t c -> b (f s) t c")
        x = r.band_split(rearrange(stft_repr, "b f t c -> b t (f c)"))
        store = [None] * len(r.layers)
        for i, block in enumerate(r.layers):
            if len(block) == 3:
                linear_t, time_t, freq_t = block
                x, ps = pack([x], "b * d")
                x, = unpack(linear_t(x), ps, "b * d")
            else:
                time_t, freq_t = block
            if r.skip_connection:
                for j in range(i):
                    x = x + store[j]
            x = self._ckpt(self._axial, x, time_t, freq_t)
            if r.skip_connection:
                store[i] = x
        return r.final_norm(x), stft_repr, window

    def _ckpt(self, fn, *args):
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    @staticmethod
    def _axial(x, time_t, freq_t):
        from einops import pack, rearrange, unpack

        x = rearrange(x, "b t f d -> b f t d")
        x, ps = pack([x], "* t d")
        x, = unpack(time_t(x), ps, "* t d")
        x = rearrange(x, "b f t d -> b t f d")
        x, ps = pack([x], "* f d")
        x, = unpack(freq_t(x), ps, "* f d")
        return x

    def forward(self, mix: torch.Tensor) -> dict[str, torch.Tensor]:
        from einops import rearrange

        r = self.r
        B, C, S = mix.shape
        x, stft_repr, window = self._features(mix)
        a, exist = self.attractors(x, self.grid(x) if self.v2 else None)
        K = a.shape[1]
        gamma, beta = self.film(a).chunk(2, dim=-1)
        xk = x[:, None] * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]  # (B, K, T, Nb, D)
        xk = xk.flatten(0, 1)
        if self.v2:
            xk = self.refiner(xk)
        mask = self._ckpt(self.head, xk)  # (B*K, T, F*C*2)
        mask = rearrange(mask.float(), "(b k) t (f c) -> b k f t c", b=B, c=2)  # complex math in fp32 under autocast
        spec = torch.view_as_complex(stft_repr.contiguous())[:, None] * torch.view_as_complex(mask.contiguous())
        spec = rearrange(spec, "b k (f s) t -> (b k s) f t", s=C)
        if r.zero_dc:
            spec = spec.clone()
            spec[:, 0] = 0.0
        wav = torch.istft(spec, **r.stft_kwargs, window=window, return_complex=False, length=S)
        return {"sources": wav.reshape(B, K, C, S), "exist_logits": exist, "attractors": a}
