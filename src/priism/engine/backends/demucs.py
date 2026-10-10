"""Demucs models (Défossez et al.), run in-process through the ``demucs`` package.

``htdemucs_ft`` is a bag of 4 fine-tuned Hybrid Transformer Demucs, one per stem (drums,
bass, other, vocals). Inference is deterministic: no random shifts (``shifts=0``), fixed
segment overlap, and the model's own weights fingerprinted in the manifest.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..manifest import ModelInfo

KNOWN = ("htdemucs", "htdemucs_ft", "htdemucs_6s")
OVERLAP = 0.25


class DemucsModel:
    def __init__(self, name: str, device: str = "cpu"):
        if name not in KNOWN:
            raise ValueError(f"unknown demucs model {name!r}; known: {', '.join(KNOWN)}")
        try:
            import demucs
            import torch
            from demucs.pretrained import get_model
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("demucs models need the [demucs] extra: pip install 'priism[demucs]'") from e
        self.name = name
        self.device = device
        self._torch = torch
        self._demucs_version = demucs.__version__
        self.model = get_model(name).to(device).eval()
        self.sample_rate = int(self.model.samplerate)
        self.stems = tuple(self.model.sources)

    def _fingerprint(self) -> str:
        """SHA-256 of the weights themselves (names, shapes, bytes), whatever cache they came from."""
        h = hashlib.sha256()
        subs = getattr(self.model, "models", [self.model])
        for sub in subs:
            for key, t in sorted(sub.state_dict().items()):
                h.update(key.encode())
                h.update(str(tuple(t.shape)).encode())
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def info(self) -> ModelInfo:
        return ModelInfo(backend="demucs", name=self.name, version=f"demucs-{self._demucs_version}",
                         sha256=self._fingerprint())

    def separate(self, audio, progress):
        from demucs.apply import apply_model

        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(audio.T))  # (2, frames)
        # The demucs CLI normalises the input by its standard deviation; same here, scaled back after.
        scale = float(x.mean(0).std()) or 1.0
        n_models = len(getattr(self.model, "models", [self.model]))
        frames = x.shape[1]

        def on_chunk(d: dict) -> None:
            if d.get("state") == "end":
                done = d.get("model_idx_in_bag", 0) + min(1.0, (d.get("segment_offset", 0) + 1) / max(frames, 1))
                progress(min(done / n_models, 1.0))

        with torch.inference_mode():
            out = apply_model(self.model, (x / scale)[None], shifts=0, split=True, overlap=OVERLAP,
                              device=self.device, callback=on_chunk)[0] * scale
        progress(1.0)
        return {stem: out[i].T.contiguous().numpy().astype(np.float32) for i, stem in enumerate(self.stems)}
