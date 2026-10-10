"""A model without ML: Butterworth band splits. For building and testing clients (MyMixxx)
without models or a GPU; the "stems" are frequency bands, not instruments.

Zero-phase filtering (``sosfiltfilt``) keeps every band aligned with the input, so the
alignment and the null test of the real pipeline can be checked with it too.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..manifest import ModelInfo

# name -> stems and their band edges in Hz (None = open)
KNOWN = {
    "bands": {"low": (None, 150.0), "mid": (150.0, 2500.0), "high": (2500.0, None)},
}
VERSION = "1"


class FakeBands:
    sample_rate = 44100

    def __init__(self, name: str = "bands"):
        if name not in KNOWN:
            raise ValueError(f"unknown fake model {name!r}; known: {', '.join(KNOWN)}")
        self.name = name
        self.bands = KNOWN[name]
        self.stems = tuple(self.bands)

    def info(self) -> ModelInfo:
        return ModelInfo(backend="fake", name=self.name, version=VERSION)

    def separate(self, audio, progress):
        out = {}
        for i, (stem, (lo, hi)) in enumerate(self.bands.items()):
            if lo is None:
                sos = butter(4, hi, btype="lowpass", fs=self.sample_rate, output="sos")
            elif hi is None:
                sos = butter(4, lo, btype="highpass", fs=self.sample_rate, output="sos")
            else:
                sos = butter(4, (lo, hi), btype="bandpass", fs=self.sample_rate, output="sos")
            out[stem] = sosfiltfilt(sos, audio, axis=0).astype(np.float32)
            progress((i + 1) / len(self.bands))
        return out
