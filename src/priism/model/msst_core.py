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


class MsstAttractorSeparator(nn.Module):
    def __init__(self, roformer: nn.Module, max_sources: int = 16, decoder_depth: int = 2, heads: int = 8,
                 init_head: int = 2, film_init_std: float = 0.01, grad_checkpoint: bool = False):
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
        a, exist = self.attractors(x)
        K = a.shape[1]
        gamma, beta = self.film(a).chunk(2, dim=-1)
        xk = x[:, None] * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]  # (B, K, T, Nb, D)
        mask = self._ckpt(self.head, xk.flatten(0, 1))  # (B*K, T, F*C*2)
        mask = rearrange(mask.float(), "(b k) t (f c) -> b k f t c", b=B, c=2)  # complex math in fp32 under autocast
        spec = torch.view_as_complex(stft_repr.contiguous())[:, None] * torch.view_as_complex(mask.contiguous())
        spec = rearrange(spec, "b k (f s) t -> (b k s) f t", s=C)
        if r.zero_dc:
            spec = spec.clone()
            spec[:, 0] = 0.0
        wav = torch.istft(spec, **r.stft_kwargs, window=window, return_complex=False, length=S)
        return {"sources": wav.reshape(B, K, C, S), "exist_logits": exist, "attractors": a}
