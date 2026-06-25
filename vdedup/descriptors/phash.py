"""Secondary visual descriptor — 64-bit DCT perceptual hash.

Cheap to compute and Hamming-compare; a coarse pre-filter and cross-check, and
the visual-matching substrate when SSCD weights are unavailable. Degrades under
strong re-encoding/crop, which is why it is secondary to the embedding.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dct

_EDGE = 32
_LOW = 8


def _phash_one(gray: np.ndarray) -> np.uint64:
    if gray.shape != (_EDGE, _EDGE):
        gray = np.asarray(Image.fromarray(gray).resize((_EDGE, _EDGE), Image.BILINEAR))
    d = dct(dct(gray.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:_LOW, :_LOW].flatten()
    med = np.median(low[1:])                # exclude DC term
    bits = low > med
    out = np.uint64(0)
    for b in bits:
        out = (out << np.uint64(1)) | np.uint64(bool(b))
    return out


def phash_frames(frames_gray: np.ndarray) -> np.ndarray:
    """frames_gray: [n, H, W] uint8 -> uint64[n]."""
    return np.array([_phash_one(f) for f in frames_gray], dtype=np.uint64)


def hamming(a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray | int:
    x = np.bitwise_xor(np.uint64(a), np.uint64(b)) if np.isscalar(a) else (a ^ b)
    if np.isscalar(x):
        return int(bin(int(x)).count("1"))
    # vectorised popcount on uint64
    x = x.astype(np.uint64)
    cnt = np.zeros_like(x, dtype=np.uint8)
    for _ in range(64):
        cnt += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return cnt
