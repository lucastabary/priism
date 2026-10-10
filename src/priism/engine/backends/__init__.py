"""Separation models behind one interface.

A model is named ``<backend>:<name>`` in profiles (``fake:bands``, ``demucs:htdemucs_ft``).
Backends import their heavy dependencies (torch, demucs) only when a model is loaded, so the
engine and the ``fake`` profile run with numpy and scipy alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from ..manifest import ModelInfo

ProgressFn = Callable[[float], None]


class Model(Protocol):
    sample_rate: int  # rate the model runs at; the engine resamples in and out
    stems: tuple[str, ...]  # stem names it returns

    def info(self) -> ModelInfo: ...

    def separate(self, audio: np.ndarray, progress: ProgressFn) -> dict[str, np.ndarray]:
        """(frames, 2) float32 at ``sample_rate`` -> {stem: (frames, 2) float32}, same length."""
        ...


BACKENDS = ("fake", "demucs")


def load_model(spec: str, device: str = "cpu") -> Model:
    backend, _, name = spec.partition(":")
    if backend == "fake":
        from .fake import FakeBands

        return FakeBands(name or "bands")
    if backend == "demucs":
        from .demucs import DemucsModel

        return DemucsModel(name, device=device)
    raise ValueError(f"unknown model backend {backend!r} in {spec!r}; known: {', '.join(BACKENDS)}")


def available_models() -> list[str]:
    """Models a profile can name; the demucs ones need the [demucs] extra."""
    from .demucs import KNOWN as DEMUCS
    from .fake import KNOWN as FAKE

    return [f"fake:{n}" for n in FAKE] + [f"demucs:{n}" for n in DEMUCS]
