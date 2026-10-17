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
# P modes. "salience": the TOP_K salience values (the original P). Their relative heights leak the partials'
# balance: a note's octave and sub-octave peaks are made of its even harmonics, so the values at f0, 2 f0 and
# f0/2 encode the even/odd profile, part of the timbre Z should carry. "binary": the same TOP_K positions at 1,
# no heights. "cancel": iterative estimation and cancellation (Klapuri 2006 style): take the most salient
# pitch, remove its harmonic bins from the spectrum, recompute, up to TOP_K notes while the next one keeps
# CANCEL_REL of the first one's salience; marked 1. It names played notes rather than strong partials.
# "bp": Basic Pitch note posteriors above its frame threshold (cleave/pitch_bp.py), the only mode that names
# the played notes reliably on the synth bench (docs/cleave-diagnostic.md 9.2); runs on CPU with onnxruntime.
P_MODES = ("salience", "binary", "cancel", "bp")
CANCEL_REL = 0.3
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

    def __init__(self, p_mode: str = "salience"):
        super().__init__()
        if p_mode not in P_MODES:
            raise ValueError(f"p_mode {p_mode!r} not in {P_MODES}")
        self.p_mode = p_mode
        self.bp = None
        if p_mode == "bp":
            from .pitch_bp import BasicPitch

            self.bp = BasicPitch()
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
        if self.p_mode == "bp":
            pitch = self._basic_pitch(wave, mag.shape[1])
        elif self.p_mode == "cancel":
            pitch = self._cancel(mag)
        else:
            sal = mag @ self.salience
            sal = sal / (sal.amax(-1, keepdim=True) + 1e-8)
            top = sal.topk(TOP_K, dim=-1)
            values = top.values if self.p_mode == "salience" else torch.ones_like(top.values)
            pitch = torch.zeros_like(sal).scatter(-1, top.indices, values)
        pitch = pitch * (level_db > SILENCE_DB).unsqueeze(-1)
        return logmel, torch.cat([pitch, loud[..., None], onset[..., None]], -1), level_db

    def _basic_pitch(self, wave: torch.Tensor, T: int) -> torch.Tensor:
        """(B, S) waveform -> (B, T, N_PITCH) 0/1 notes from Basic Pitch, padded or cut to the T STFT frames."""
        from .pitch_bp import FRAME_THRESHOLD, MIDI_LO as BP_LO

        post = self.bp.notes(wave.float().cpu().numpy(), SR)[:, :, MIDI_LO - BP_LO: MIDI_LO - BP_LO + N_PITCH]
        notes = torch.from_numpy((post > FRAME_THRESHOLD).astype(np.float32))[:, :T]
        return F.pad(notes, (0, 0, 0, T - notes.shape[1])).to(wave.device)

    def _cancel(self, mag: torch.Tensor) -> torch.Tensor:
        """(B, T, bins) magnitudes -> (B, T, N_PITCH) 0/1 notes by estimation and cancellation."""
        harm = (self.salience > 0).float()  # (bins, N_PITCH): the bins each pitch's harmonics touch
        rest = mag.clone()
        pitch = torch.zeros(*mag.shape[:2], N_PITCH, device=mag.device)
        first = None
        for _ in range(TOP_K):
            sal = rest @ self.salience
            best, idx = sal.max(-1)
            if first is None:
                first = best
            keep = (best > CANCEL_REL * first) & (best > 0)
            pitch.scatter_(-1, idx[..., None], keep[..., None].float())
            rest = rest * (1 - harm[:, idx].permute(1, 2, 0) * keep[..., None].float())
        return pitch
