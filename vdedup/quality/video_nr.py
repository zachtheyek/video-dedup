"""Stage 8 (video) — no-reference video quality, measured from decoded signal.

None of these trust container claims. R_eff estimates *true* detail spectrally so
an upscaled-then-re-encoded file cannot advertise resolution it does not deliver.
"""
from __future__ import annotations

import numpy as np


def radial_psd_detail(gray_frames: np.ndarray, band: tuple[float, float] = (0.25, 0.9)) -> float:
    """Detail index delta in [0,1]: fraction of radially-averaged power-spectrum
    energy in a high-spatial-frequency band. Genuine high-res content carries
    energy out to high frequencies; upscaled/blurred content rolls off early."""
    if gray_frames.shape[0] == 0:
        return 0.0
    h, w = gray_frames.shape[1:3]
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2)  # 0..1 normalised radius
    deltas = []
    for fr in gray_frames:
        f = np.fft.fftshift(np.fft.fft2(fr.astype(np.float64)))
        psd = np.abs(f) ** 2
        total = psd.sum() + 1e-9
        band_mask = (r >= band[0]) & (r <= band[1])
        deltas.append(float(psd[band_mask].sum() / total))
    return float(np.median(deltas))


def r_eff(claimed_w: int, claimed_h: int, gray_frames: np.ndarray,
          band: tuple[float, float] = (0.25, 0.9),
          detail_ref: float | None = None) -> tuple[float, float]:
    """Return (R_eff, delta).

    With `detail_ref` given, R_eff = claimed_pixels * clip(delta/detail_ref, 0, 2)
    — i.e. claimed pixels scaled by how close the measured detail is to full-
    detail content (penalising upscales/blur on a pixel-comparable scale). With
    detail_ref None, R_eff = claimed_pixels * delta (raw)."""
    delta = radial_psd_detail(gray_frames, band)
    if detail_ref:
        factor = min(2.0, max(0.0, delta / detail_ref))
    else:
        factor = delta
    return float(claimed_w * claimed_h * factor), delta


def bpp_norm(bitrate_bps: float | None, w: int, h: int, fps: float | None,
             vcodec: str | None, codec_eff: dict[str, float]) -> float | None:
    """Bits per pixel per frame, normalised by codec efficiency (so HEVC/AV1 are
    compared fairly against H.264). None if bitrate/fps unknown."""
    if not bitrate_bps or not fps or fps <= 0 or w <= 0 or h <= 0:
        return None
    bpp = bitrate_bps / (w * h * fps)
    eff = codec_eff.get((vcodec or "").lower(), 1.0)
    return float(bpp * eff)


def blockiness(gray_frames: np.ndarray, block: int = 8) -> float:
    """Mean gradient discontinuity at block boundaries vs interior — a DCT-block
    artifact proxy. Higher = more blocking. Returns ~0 for clean content."""
    if gray_frames.shape[0] == 0:
        return 0.0
    vals = []
    for fr in gray_frames.astype(np.float64):
        gx = np.abs(np.diff(fr, axis=1))
        h, w = gx.shape
        cols = np.arange(w)
        on = gx[:, (cols % block) == (block - 1)].mean() if w >= block else 0.0
        off = gx[:, (cols % block) != (block - 1)].mean() if w >= block else 1.0
        vals.append((on - off) / (off + 1e-6))
    return float(max(0.0, np.median(vals)))


def banding(gray_frames: np.ndarray) -> float:
    """Banding proxy: in smooth regions, real content has small continuous
    gradients; banded content has many exactly-flat steps. Higher = more banding."""
    if gray_frames.shape[0] == 0:
        return 0.0
    vals = []
    for fr in gray_frames.astype(np.int16):
        g = np.abs(np.diff(fr, axis=1))
        smooth = g < 4
        flat = (g == 0) & smooth
        denom = smooth.sum() + 1e-6
        vals.append(flat.sum() / denom)
    return float(np.median(vals))
