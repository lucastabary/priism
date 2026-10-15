"""Deterministic front end of Cleave: the log-mel input, and the P and L factors given to the predictor.

Nothing here is learned, so P and L tell the predictor *what* is played (notes, onsets, intensity) without
saying *how* it sounds; the timbre has to come through Z. Two choices keep timbre out of them:

- L is one broadband A-weighted level per frame. Per-band levels would hand over the spectral envelope.
- P is a harmonic salience on a semitone grid, normalised per frame (no absolute level) and cut to its
  ``TOP_K`` strongest pitches: which notes, barely how their partials are balanced.

Everything runs on the device of the waveform, from one STFT.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 128
F_MIN, F_MAX = 20.0, 16000.0
MIDI_LO, N_PITCH, N_HARM, HARM_DECAY, TOP_K = 24, 84, 6, 0.8, 4
SILENCE_DB = -60.0  # frames below this level carry no pitch; chunks never above it are silent
PL_DIM = N_PITCH + 2  # salience, loudness, onset strength


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = F_MIN,
                   fmax: float = F_MAX) -> np.ndarray:
    """(bins, n_mels) triangles of unit sum, so a band is the mean power under it. The lowest bands are
    narrower than an FFT bin and would catch nothing: they take their nearest bin instead."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((len(freqs), n_mels), np.float32)
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[:, i] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
        if fb[:, i].sum() == 0:
            fb[np.argmin(np.abs(freqs - mid)), i] = 1.0
    return fb / fb.sum(0, keepdims=True)


def a_weighting(freqs: np.ndarray) -> np.ndarray:
    """Power gain of the A-weighting curve (IEC 61672), 0 dB at 1 kHz."""
    f2 = np.asarray(freqs, np.float64) ** 2
    ra = (12194.0**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    return ((ra * 10 ** (2.0 / 20)) ** 2).astype(np.float32)


def salience_matrix(sr: int = SR, n_fft: int = N_FFT) -> np.ndarray:
    """(bins, N_PITCH): pitch p sums the magnitudes at its first harmonics, weighted HARM_DECAY^(h-1), each
    harmonic split linearly between its two nearest bins. Low pitches are coarse at this FFT size (bins of
    21.5 Hz); their upper harmonics still tell them apart."""
    bin_hz = sr / n_fft
    m = np.zeros((n_fft // 2 + 1, N_PITCH), np.float32)
    for p in range(N_PITCH):
        f0 = 440.0 * 2 ** ((MIDI_LO + p - 69) / 12)
        for h in range(1, N_HARM + 1):
            x = h * f0 / bin_hz
            i = int(x)
            if i + 1 >= m.shape[0]:
                break
            w = x - i
            m[i, p] += (1 - w) * HARM_DECAY ** (h - 1)
            m[i + 1, p] += w * HARM_DECAY ** (h - 1)
    return m


class Frontend(nn.Module):
    """(B, S) mono waveform at SR -> log-mel (B, T, N_MELS), P/L factors (B, T, PL_DIM), level (B, T) in dB.

    T = S // HOP + 1 (centred STFT). The log-mel is scaled so -100..-10 dB maps to about -2..1."""

    def __init__(self):
        super().__init__()
        freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("mel_fb", torch.from_numpy(mel_filterbank()), persistent=False)
        self.register_buffer("a_weight", torch.from_numpy(a_weighting(freqs)), persistent=False)
        self.register_buffer("salience", torch.from_numpy(salience_matrix()), persistent=False)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = torch.stft(wave.float(), N_FFT, HOP, window=self.window, center=True, return_complex=True)
        mag = spec.abs().transpose(1, 2) / self.window.sum()  # (B, T, bins); a full-scale sine peaks near 0.5
        power = mag**2
        logmel = (10 * torch.log10(power @ self.mel_fb + 1e-10) + 40) / 30
        level_db = 10 * torch.log10((power * self.a_weight).sum(-1) + 1e-10)
        loud = level_db.clamp(-80, 0) / 80
        flux = F.pad(torch.relu(logmel[:, 1:] - logmel[:, :-1]).mean(-1), (1, 0))
        onset = torch.tanh(4 * flux)
        sal = mag @ self.salience
        sal = sal / (sal.amax(-1, keepdim=True) + 1e-8)
        top = sal.topk(TOP_K, dim=-1)
        pitch = torch.zeros_like(sal).scatter(-1, top.indices, top.values)
        pitch = pitch * (level_db > SILENCE_DB).unsqueeze(-1)
        return logmel, torch.cat([pitch, loud[..., None], onset[..., None]], -1), level_db
