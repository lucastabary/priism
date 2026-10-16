"""Cascades of pretrained separators on the SW stems (option B of ``docs/cleave-diagnostic.md``).

Facet splits each BS-Roformer-SW stem with masks, so what SW puts in a stem bounds what Facet can get out of
it. A cascade asks whether finer pretrained models do better: SW first (the model best at the 6 stems), then
a specialist on one of its stems, e.g. a DrumSep model on the drums stem (kick, snare, toms, hi-hat, ride,
crash). As in the engine's profiles, every step is subtractive: the specialist's outputs plus the stem's
*rest* (stem minus their sum) give back the stem exactly, so nothing is created or lost.

``MsstSplitter`` runs any MSST checkpoint (ZFTurbo's Music-Source-Separation-Training) of the types below,
chunked with MSST's own overlap-add (``utils.model_utils.demix``, generic mode: linear fades of a tenth of
the chunk, reflect padding at the song's edges), but without importing MSST's training utilities, which need
``ml_collections``: only the model classes are imported.
"""

from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MODEL_TYPES = ("mdx23c", "bs_roformer", "mel_band_roformer")


class _Cfg(dict):
    """Attribute access over a nested dict: the part of ``ml_collections.ConfigDict`` MSST models use."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return _Cfg(v) if isinstance(v, dict) and not isinstance(v, _Cfg) else v


def _target_instruments(config) -> list[str]:  # MSST's utils.model_utils.prefer_target_instrument
    t = config.training.get("target_instrument")
    return [t] if t else list(config.training.instruments)


def _msst_import(msst_path: str | Path, model_type: str):
    if str(msst_path) not in sys.path:
        sys.path.insert(0, str(msst_path))
    if model_type == "mdx23c":
        try:
            import utils.model_utils  # noqa: F401  (needs ml_collections)
        except ImportError:  # the model file only needs this one helper from it
            stub = types.ModuleType("utils.model_utils")
            stub.prefer_target_instrument = _target_instruments
            sys.modules["utils.model_utils"] = stub
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net  # type: ignore

        return lambda cfg: TFC_TDF_net(cfg)
    if model_type == "bs_roformer":
        from models.bs_roformer.bs_roformer import BSRoformer  # type: ignore

        return lambda cfg: BSRoformer(**dict(cfg["model"]))
    if model_type == "mel_band_roformer":
        from models.bs_roformer.mel_band_roformer import MelBandRoformer  # type: ignore

        return lambda cfg: MelBandRoformer(**dict(cfg["model"]))
    raise ValueError(f"unknown MSST model type {model_type!r} (known: {MODEL_TYPES})")


class MsstSplitter:
    """One MSST checkpoint as a function (C, S) waveform at ``sample_rate`` -> (N, C, S) estimates of ``stems``."""

    def __init__(self, model_type: str, config: str | Path, ckpt: str | Path, msst_path: str | Path,
                 device: str = "cuda"):
        import yaml

        cfg = _Cfg(yaml.load(Path(config).read_text(), Loader=yaml.FullLoader))
        build = _msst_import(msst_path, model_type)
        model = build(cfg)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
        model.load_state_dict(state)
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device
        self.name = Path(ckpt).name
        self.stems = tuple(_target_instruments(cfg))
        self.sample_rate = int(cfg.audio.get("sample_rate", 44100))
        inf = cfg.get("inference", {}) or {}
        self.chunk = int(inf.get("chunk_size") or cfg.audio.chunk_size)
        self.overlap = int(inf.get("num_overlap", 2))
        self.batch = int(inf.get("batch_size", 1))
        self.amp = bool(cfg.training.get("use_amp", True))

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(wav, np.float32))
        return chunked(lambda b: self._forward(b), x, len(self.stems), self.chunk, self.overlap, self.batch).numpy()

    def _forward(self, b: torch.Tensor) -> torch.Tensor:
        cuda = self.device.startswith("cuda")
        with torch.autocast("cuda", torch.float16, enabled=cuda and self.amp):
            y = self.model(b.to(self.device))
        y = y.float().cpu()
        return y[:, None] if y.dim() == 3 else y  # a one-stem model may return (B, C, S)


def chunked(forward, mix: torch.Tensor, n_out: int, chunk: int, overlap: int, batch: int = 1) -> torch.Tensor:
    """MSST's generic overlap-add (``demix``): chunks of ``chunk`` samples every ``chunk // overlap``, linear
    fades of ``chunk // 10`` (none at the song's ends), reflect padding of ``chunk - step`` at both ends when
    the song is long enough. ``forward``: (B, C, chunk) -> (B, n_out, C, chunk). Returns (n_out, C, S)."""
    S = mix.shape[-1]
    step = chunk // overlap
    border = chunk - step
    fade = chunk // 10
    padded = S > 2 * border and border > 0
    x = torch.nn.functional.pad(mix[None], (border, border), mode="reflect")[0] if padded else mix
    L = x.shape[-1]
    window = torch.ones(chunk)
    window[:fade] = torch.linspace(0, 1, fade)
    window[-fade:] = torch.linspace(1, 0, fade)
    out = torch.zeros(n_out, *x.shape)
    count = torch.zeros(L)
    starts = list(range(0, L, step))
    for k in range(0, len(starts), batch):
        group = starts[k:k + batch]
        parts = []
        for i in group:
            p = x[:, i:i + chunk]
            n = p.shape[-1]
            mode = "reflect" if n > chunk // 2 else "constant"
            parts.append(torch.nn.functional.pad(p, (0, chunk - n), mode=mode))
        y = forward(torch.stack(parts))
        for j, i in enumerate(group):
            n = min(chunk, L - i)
            w = window.clone()
            if i == 0:
                w[:fade] = 1
            if i + step >= L:
                w[-fade:] = 1
            out[..., i:i + n] += y[j, ..., :n] * w[:n]
            count[i:i + n] += w[:n]
    out /= count
    return out[..., border:-border] if padded else out


@dataclass
class CascadeStep:
    """Split SW stem ``stem`` with ``splitter``; ``merge`` names groups of its outputs scored as one source
    in a second variant (e.g. ``{"cymbals": ["ride", "crash"]}``)."""
    stem: str
    splitter: object
    merge: dict = field(default_factory=dict)


def split(step: CascadeStep, audio: np.ndarray, sample_rate: int) -> dict[bool, tuple[list[str], list[np.ndarray]]]:
    """(C, S) stem at ``sample_rate`` -> {merged: (names, waveforms)}, one splitter pass: its outputs, with
    the ``merge`` groups summed under ``True`` (only when the step has merges), plus ``<stem>_rest`` = stem
    minus their sum, so each variant's outputs sum to the stem exactly."""
    sp = step.splitter
    y = _resample(np.asarray(sp(_resample(audio, sample_rate, sp.sample_rate)), np.float32), sp.sample_rate,
                  sample_rate)
    n = audio.shape[-1]
    y = y[..., :n] if y.shape[-1] >= n else np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, n - y.shape[-1])])
    rest = audio.astype(np.float32) - y.sum(0)  # the same for every variant: merging keeps the sum
    variants = {False: (list(sp.stems), list(y))}
    if step.merge:
        names = list(sp.stems)
        grouped = {m for g in step.merge.values() for m in g}
        keep = [(nm, o) for nm, o in zip(names, y) if nm not in grouped]
        keep += [(g, y[[names.index(m) for m in members]].sum(0)) for g, members in step.merge.items()]
        variants[True] = ([k for k, _ in keep], [o for _, o in keep])
    return {v: ([f"{step.stem}_{nm}" for nm in names] + [f"{step.stem}_rest"], outs + [rest])
            for v, (names, outs) in variants.items()}


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1).astype(np.float32)


def parse_steps(specs: list[str], merges: list[str], msst_path: str | Path, device: str) -> list[CascadeStep]:
    """``STEM=TYPE:CONFIG:CKPT`` (one per split SW stem) and ``NAME=a+b`` merges -> steps (models loaded)."""
    merge = {}
    for m in merges or []:
        name, members = m.split("=", 1)
        merge[name] = members.split("+")
    steps = []
    for s in specs or []:
        stem, rest = s.split("=", 1)
        model_type, config, ckpt = rest.split(":", 2)
        t = time.time()
        sp = MsstSplitter(model_type, config, ckpt, msst_path, device)
        print(f"cascade: {stem} -> {model_type} {Path(ckpt).name} {sp.stems} ({time.time() - t:.0f} s to load)",
              flush=True)
        steps.append(CascadeStep(stem, sp, {k: v for k, v in merge.items() if set(v) <= set(sp.stems)}))
    return steps
