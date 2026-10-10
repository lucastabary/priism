"""Degradations applied to the mix only, at training time: targets stay clean.

Real songs reach a DJ as MP3/AAC/Opus (SoundCloud streams, shop downloads), so
the separator must learn to separate through codec artefacts.
"""

from __future__ import annotations

import subprocess

import numpy as np

CODECS = {"mp3": ("libmp3lame", "mp3"), "aac": ("aac", "adts"), "opus": ("libopus", "ogg")}


def lossy(mix: np.ndarray, sr: int, codec: str = "mp3", kbps: int = 192) -> np.ndarray:
    """Encode and decode a stereo float mix with ffmpeg; same length and rate as the input."""
    enc, fmt = CODECS[codec]
    pcm = np.ascontiguousarray(mix, dtype=np.float32).tobytes()
    rate = 48000 if codec == "opus" else sr  # opus only runs at 48 kHz
    encode = ["ffmpeg", "-loglevel", "error", "-threads", "1", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
              "-ar", str(rate), "-c:a", enc, "-b:a", f"{kbps}k", "-threads", "1", "-f", fmt, "-"]
    coded = subprocess.run(encode, input=pcm, capture_output=True, check=True).stdout
    decode = ["ffmpeg", "-loglevel", "error", "-threads", "1", "-i", "-", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-"]
    out = np.frombuffer(subprocess.run(decode, input=coded, capture_output=True, check=True).stdout, np.float32)
    out = out.reshape(-1, 2)
    # Encoders add a few ms of priming delay: align on the cross-correlation peak, then trim/pad.
    probe = min(len(mix), sr * 2)
    a = mix[:probe, 0].astype(np.float64)
    b = out[: probe + 4096, 0].astype(np.float64)
    corr = np.correlate(b, a[: probe - 4096] if probe > 8192 else a, mode="valid") if probe > 0 else np.zeros(1)
    lag = int(np.argmax(corr)) if len(corr) else 0
    out = out[lag:]
    if len(out) < len(mix):
        out = np.pad(out, ((0, len(mix) - len(out)), (0, 0)))
    return out[: len(mix)]


def random_lossy(mix: np.ndarray, sr: int, rng: np.random.Generator, p: float = 0.4) -> tuple[np.ndarray, dict]:
    """With probability ``p``, a random codec and bitrate; returns the mix and what was applied."""
    if rng.random() >= p:
        return mix, {}
    codec = str(rng.choice(list(CODECS), p=[0.6, 0.25, 0.15]))
    kbps = int(rng.choice([96, 128, 160, 192, 256, 320]))
    return lossy(mix, sr, codec, kbps), {"codec": codec, "kbps": kbps}
