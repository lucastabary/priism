"""Tempo and key of real songs, to pair them for MixIT.

MixIT mixes two real songs and asks the model to split the result. If the two
songs differ in tempo or key, the model can cheat by separating along those cues
instead of learning sources. Songs are therefore paired only when a small
time-stretch and pitch-shift can align them. Tempo is plain numpy/scipy (right on
16 of 16 generated songs, up to half/double); the key comes from Essentia.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _spectrogram(mono: np.ndarray, sr: int, n: int = 4096) -> tuple[np.ndarray, np.ndarray, float]:
    f, _, z = stft(mono, sr, nperseg=n, noverlap=n - n // 8, boundary=None, padded=False)
    return f, np.abs(z), sr / (n // 8)


def estimate_tempo(mono: np.ndarray, sr: int, lo: float = 70.0, hi: float = 180.0, prior_bpm: float = 128.0
                   ) -> float:
    """BPM from the autocorrelation of spectral flux, with a mild preference for dance tempos."""
    f, mag, fps = _spectrogram(mono, sr, 2048)
    logm = np.log1p(100 * mag)
    flux = np.maximum(np.diff(logm, axis=1), 0).sum(0)
    flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")  # remove slow loudness changes
    flux = np.maximum(flux, 0)
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    lags = np.arange(len(ac))
    ok = (lags >= fps * 60 / hi) & (lags <= fps * 60 / lo)
    bpm = 60 * fps / np.maximum(lags, 1)
    # Score each candidate with its multiples (a beat lag also correlates at 2x and 4x).
    score = np.zeros_like(ac)
    for m, w in ((1, 1.0), (2, 0.5), (4, 0.25)):
        idx = np.minimum(lags * m, len(ac) - 1)
        score += w * ac[idx]
    score *= np.exp(-0.5 * (np.log2(bpm / prior_bpm) / 1.0) ** 2)
    score[~ok] = -np.inf
    k = int(np.argmax(score))
    if 0 < k < len(ac) - 1 and np.isfinite(score[k - 1]) and np.isfinite(score[k + 1]):  # parabolic refinement
        a, b, c = score[k - 1], score[k], score[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    return float(60 * fps / k)


def estimate_key(path: str, sample_rate: int = 44100) -> tuple[int, str, float]:
    """(tonic pitch class, 'major' | 'minor', strength) with Essentia's EDM key profile.

    A home-made chroma + Krumhansl estimate did no better than Essentia on generated songs
    (both near 3 songs in 4 with the right scale), and Essentia's ``edma`` profile was tuned
    on electronic music, so it is used as is (``pip install essentia``).
    """
    import essentia.standard as es

    audio = es.MonoLoader(filename=str(path), sampleRate=sample_rate)()
    key, scale, strength = es.KeyExtractor(profileType="edma", sampleRate=sample_rate)(audio)
    key = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}.get(key, key)
    return NAMES.index(key), scale, float(strength)


def minor_tonic(tonic: int, mode: str) -> int:
    """A major key and its relative minor share their notes: compare keys on the minor tonic."""
    return tonic % 12 if mode == "minor" else (tonic - 3) % 12


def pairing(a: dict, b: dict, max_shift: int = 2, max_stretch: float = 0.06) -> dict | None:
    """How to align song ``b`` on song ``a`` ({bpm, tonic, mode} each), or None if too far.

    Returns ``{"semitones": s, "stretch": r}``: pitch-shift ``b`` by s semitones and change
    its tempo by the factor r (r > 1 is faster). Half and double tempos count as the same.
    """
    best_r = None
    for m in (0.5, 1.0, 2.0):
        r = a["bpm"] / (b["bpm"] * m)
        if abs(r - 1) <= max_stretch and (best_r is None or abs(r - 1) < abs(best_r - 1)):
            best_r = r * m  # the factor applied to b's real tempo
    if best_r is None:
        return None
    d = (minor_tonic(a["tonic"], a["mode"]) - minor_tonic(b["tonic"], b["mode"])) % 12
    s = d if d <= 6 else d - 12
    if abs(s) > max_shift:
        return None
    return {"semitones": int(s), "stretch": float(best_r)}


def describe(path: str, sample_rate: int = 44100, start_s: float = 30.0, duration_s: float = 60.0) -> dict:
    """Tempo (on an excerpt) and key (on the whole file) of one audio file."""
    import subprocess

    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start_s), "-i", str(path), "-t", str(duration_s),
                          "-map", "0:a:0", "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
                         capture_output=True, check=True).stdout
    mono = np.frombuffer(raw, dtype=np.float32)
    if len(mono) < 10 * sample_rate:  # short file: start from the beginning
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-ac", "1",
                              "-ar", str(sample_rate), "-"], capture_output=True, check=True).stdout
        mono = np.frombuffer(raw, dtype=np.float32)
    tonic, mode, strength = estimate_key(path, sample_rate)
    return {"bpm": round(estimate_tempo(mono, sample_rate), 2), "tonic": tonic, "mode": mode,
            "key": NAMES[tonic] + ("m" if mode == "minor" else ""), "key_strength": round(strength, 3)}


def find_pairs(tags: dict[str, dict], max_shift: int = 2, max_stretch: float = 0.06) -> list[tuple[str, str, dict]]:
    """Every pair of songs that a small pitch-shift and time-stretch can align."""
    names = sorted(tags)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = pairing(tags[a], tags[b], max_shift, max_stretch)
            if p:
                out.append((a, b, p))
    return out
