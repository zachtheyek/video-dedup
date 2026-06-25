"""Frame filtering — the single most important precision filter.

Low-entropy frames (fades to black/white, solid title cards, plain credits) match
*everything* and are the dominant source of false cluster edges. Compute per-frame
Shannon entropy of the luminance histogram (plus an edge-density check) and
exclude frames below threshold from indexing and matching.
"""
from __future__ import annotations

import numpy as np


def frame_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def _edge_density(gray: np.ndarray) -> float:
    gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
    gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
    return float((gx > 16).mean() + (gy > 16).mean()) / 2.0


def informative_mask(frames_gray: np.ndarray, entropy_min: float = 4.0,
                     edge_density_min: float = 0.01) -> np.ndarray:
    """[n,H,W] uint8 -> bool[n], True where the frame carries discriminative content."""
    n = frames_gray.shape[0]
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        g = frames_gray[i]
        if frame_entropy(g) >= entropy_min and _edge_density(g) >= edge_density_min:
            out[i] = True
    return out
