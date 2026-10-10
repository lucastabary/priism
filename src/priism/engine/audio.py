"""Reading the input track and writing lossless stems.

Frame 0 of every stem must be frame 0 of the track *as Mixxx decodes it*. Mixxx's decoders
do not agree with other decoders (its MP3 decoder skips the first MP3 frame and ignores the
LAME encoder delay), so MyMixxx decodes the track itself and hands priism a PCM WAV. priism
then reads that WAV as it is: no resampling, no trimming. Other formats are accepted for
standalone use, decoded by libsndfile or ffmpeg, but their timeline is not guaranteed to
match Mixxx's, which the manifest records in ``source.decoder``.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Formats libsndfile reads with an exact, untouched timeline.
PCM_EXTS = {".wav", ".wave", ".flac", ".aif", ".aiff"}
FLAC_MAX = 1.0 - 2.0**-23  # largest value a 24-bit FLAC sample holds


@dataclass
class Track:
    audio: np.ndarray  # (frames, 2) float32; mono inputs are copied to both channels
    sample_rate: int
    channels: int  # channels of the input file (1 or 2)
    decoder: str

    @property
    def frames(self) -> int:
        return len(self.audio)


def read_track(path: str | Path) -> Track:
    path = Path(path)
    if path.suffix.lower() in PCM_EXTS:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        decoder = f"priism:libsndfile-{sf.__libsndfile_version__}"
    else:
        x, sr = _ffmpeg_decode(path)
        decoder = "priism:ffmpeg"
    if x.shape[1] not in (1, 2):
        raise ValueError(f"{path}: {x.shape[1]} channels; only mono and stereo are supported")
    channels = x.shape[1]
    if channels == 1:
        x = np.repeat(x, 2, axis=1)
    if len(x) == 0:
        raise ValueError(f"{path}: no audio")
    return Track(np.ascontiguousarray(x, dtype=np.float32), int(sr), channels, decoder)


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(f"{path.suffix} input needs ffmpeg and ffprobe on PATH (or pass a decoded WAV)")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=sample_rate,channels", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    sr, ch = int(probe[0]), min(int(probe[1]), 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le",
                          "-ac", str(ch), "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).copy(), sr


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """(frames, C) -> (frames', C) with a zero-phase polyphase filter (no time shift)."""
    if sr_from == sr_to:
        return x
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def fit(x: np.ndarray, frames: int) -> np.ndarray:
    """Trim or zero-pad along time to exactly ``frames``."""
    if len(x) >= frames:
        return x[:frames]
    return np.concatenate([x, np.zeros((frames - len(x), x.shape[1]), dtype=x.dtype)])


def quantize24(x: np.ndarray) -> np.ndarray | None:
    """The values a 24-bit FLAC will give back, or None when ``x`` would clip."""
    if np.max(np.abs(x), initial=0.0) > FLAC_MAX:
        return None
    return (np.round(x.astype(np.float64) * 2.0**23) / 2.0**23).astype(np.float32)


def write_lossless(dest_no_ext: Path, x: np.ndarray, sample_rate: int) -> tuple[Path, np.ndarray]:
    """Write a stem losslessly and return (file, the exact samples a reader will get back).

    24-bit FLAC when the signal fits in full scale, 32-bit float WAV otherwise (a separated
    stem can peak above 0 dBFS even when the mix does not, and clipping it would break the
    sum back to the mix). Returning the stored samples lets the caller compute the residual
    from what is on disk, so stems + residual decode to the source exactly.
    """
    q = quantize24(x)
    if q is not None:
        path = dest_no_ext.with_suffix(".flac")
        sf.write(str(path), q, sample_rate, format="FLAC", subtype="PCM_24")
        return path, q
    path = dest_no_ext.with_suffix(".wav")
    x = x.astype(np.float32)
    sf.write(str(path), x, sample_rate, format="WAV", subtype="FLOAT")
    return path, x


def write_residual(dest_no_ext: Path, x: np.ndarray, sample_rate: int) -> Path:
    """Like a stem. Since the stems are stored exactly, the only error of stems + residual
    against the source is the residual's 24-bit rounding: at most 2**-24 (-144 dBFS) per
    sample, and none at all for a 16- or 24-bit PCM source. A float WAV would be exact for
    a float source too, at four times the size of a FLAC."""
    return write_lossless(dest_no_ext, x, sample_rate)[0]
