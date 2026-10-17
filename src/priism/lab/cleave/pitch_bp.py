"""Basic Pitch (Bittner et al., ICASSP 2022, Spotify, Apache 2.0) as Cleave's P factor: frame-level note
posteriors on 88 keys, run from its ONNX export with onnxruntime (no TensorFlow).

The hand-made salience P mostly returned partials, not notes (docs/cleave-diagnostic.md 9.2: on held synth
notes and chords, precision 0.29 and 0 % exact frames, against 0.87 and 61 % for Basic Pitch at 0.3), and its
values leaked the harmonic profile that Z should carry.

Basic Pitch reads 22.05 kHz audio with a hop of 256: 86.13 frames per second, the same rate as the front end
(44.1 kHz, hop 512), so frame i of one is frame i of the other. The model file comes from
``PRIISM_BASIC_PITCH_ONNX``, else ``data/models/basic-pitch/nmp.onnx``, else an installed ``basic_pitch``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BP_SR, BP_HOP = 22050, 256
WINDOW = BP_SR * 2 - BP_HOP  # the model's fixed input length (43844 samples, 172 frames)
N_OVERLAP = 30  # frames shared by neighbouring windows, half of them dropped on each side (as basic_pitch)
MIDI_LO = 21  # the 88 keys
FRAME_THRESHOLD = 0.3  # basic_pitch's default frame threshold, also the best one on the synth bench


def model_path() -> Path:
    env = os.environ.get("PRIISM_BASIC_PITCH_ONNX")
    if env:
        return Path(env)
    local = Path("data/models/basic-pitch/nmp.onnx")
    if local.exists():
        return local
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError as e:
        raise FileNotFoundError("Basic Pitch ONNX model not found: set PRIISM_BASIC_PITCH_ONNX") from e
    return Path(ICASSP_2022_MODEL_PATH).with_name("nmp.onnx")


class BasicPitch:
    def __init__(self, path: str | Path | None = None, threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path or model_path()), opts, providers=["CPUExecutionProvider"])

    def notes(self, wave: np.ndarray, sr: int = 44100) -> np.ndarray:
        """(B, S) mono at ``sr`` (a multiple of 22050) -> (B, ceil-ish S / (sr / 86.13), 88) note posteriors."""
        from scipy.signal import resample_poly

        x = resample_poly(np.asarray(wave, np.float32), 1, sr // BP_SR, axis=-1).astype(np.float32)
        overlap = N_OVERLAP * BP_HOP
        hop = WINDOW - overlap
        x = np.pad(x, ((0, 0), (overlap // 2, 0)))
        n = x.shape[1]
        starts = list(range(0, n, hop))
        wins = np.zeros((len(x) * len(starts), WINDOW), np.float32)
        for b in range(len(x)):
            for j, s in enumerate(starts):
                seg = x[b, s:s + WINDOW]
                wins[b * len(starts) + j, :len(seg)] = seg
        out = self.session.run(["StatefulPartitionedCall:1"], {"serving_default_input_2:0": wins[..., None]})[0]
        h = N_OVERLAP // 2
        out = out[:, h:-h].reshape(len(x), -1, out.shape[-1])
        n_frames = int(np.floor((n - overlap // 2) * 86 / BP_SR))  # basic_pitch trims with its integer fps
        return out[:, :n_frames]
