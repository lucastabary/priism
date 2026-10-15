"""Mask sampling for the Cleave JEPA: whole sections of the song, and frequency bands inside visible chunks.

Section masks (whole chunks, runs of a few) make the model recall how a source sounds elsewhere in the
song: the reason for having the whole song in context. Band masks (a few rows of patches over a visible
chunk) make it predict partials from the timbre, i.e. learn the harmonic envelope. Silent chunks are never
targets: there is nothing to predict there.
"""

from __future__ import annotations

import numpy as np


def sample_mask(active: np.ndarray, n_tcol: int, n_rows: int, rng: np.random.Generator,
                section_frac: tuple[float, float] = (0.25, 0.4), section_len: tuple[int, int] = (1, 4),
                band_p: float = 0.5, band_rows: tuple[int, int] = (2, 4)) -> np.ndarray:
    """(C,) active chunks -> (C, n_tcol * n_rows) bool patch mask (True = masked, a target).

    Patch index = tcol * n_rows + row. At least one active chunk always stays (partly) visible, so the
    global encoder has something to read."""
    C = len(active)
    mask = np.zeros((C, n_tcol, n_rows), bool)
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return mask.reshape(C, -1)
    goal = int(round(rng.uniform(*section_frac) * len(idx)))
    goal = min(goal, len(idx) - 1)
    full = np.zeros(C, bool)
    tries = 0
    while full.sum() < goal and tries < 100:
        tries += 1
        start = rng.choice(idx)
        for c in range(start, min(C, start + rng.integers(section_len[0], section_len[1] + 1))):
            if active[c] and full.sum() < goal:
                full[c] = True
    mask[full] = True
    for c in idx[~full[idx]]:
        if rng.random() < band_p:
            r = rng.integers(band_rows[0], band_rows[1] + 1)
            r0 = rng.integers(0, n_rows - r + 1)
            mask[c, :, r0:r0 + r] = True
    return mask.reshape(C, -1)
