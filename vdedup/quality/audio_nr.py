"""Stage 8 (audio) — no-reference audio quality, measured from decoded signal.

audio_bandwidth (spectral rolloff) is the audio analogue of R_eff and the single
most diagnostic signal: lossy encoding imposes a hard low-pass cutoff regardless
of the bitrate the container claims.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def audio_bandwidth(samples: np.ndarray, sr: int, rolloff: float = 0.985) -> float:
    """Spectral rolloff frequency (Hz): the frequency below which `rolloff` of the
    total spectral energy lies. A transcoded/low-bitrate track shows a hard cutoff
    well below sr/2."""
    if samples.size < 2048:
        return 0.0
    f, pxx = signal.welch(samples, fs=sr, nperseg=min(4096, samples.size))
    csum = np.cumsum(pxx)
    if csum[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(csum, rolloff * csum[-1]))
    idx = min(idx, len(f) - 1)
    return float(f[idx])


def abr_norm(bitrate_bps: float | None, channels: int | None, acodec: str | None,
             codec_eff: dict[str, float]) -> float | None:
    """Bitrate per channel, codec-efficiency-normalised (Opus/AAC > MP3 per bit)."""
    if not bitrate_bps or not channels or channels <= 0:
        return None
    eff = codec_eff.get((acodec or "").lower(), 1.0)
    return float((bitrate_bps / channels) * eff)


def clip_ratio(samples: np.ndarray, thresh: float = 0.999) -> float:
    """Fraction of samples at (near) full scale — a clipping/distortion proxy."""
    if samples.size == 0:
        return 0.0
    peak = np.max(np.abs(samples)) + 1e-9
    return float((np.abs(samples) >= thresh * peak).mean())


def dropout_ratio(samples: np.ndarray, sr: int, win_ms: float = 20.0,
                  silence_db: float = -60.0) -> float:
    """Fraction of short interior windows that are effectively silent — a dropout
    proxy. Leading/trailing silence is ignored (not a dropout)."""
    if samples.size < sr // 10:
        return 0.0
    win = max(1, int(sr * win_ms / 1000))
    n = samples.size // win
    if n < 3:
        return 0.0
    frames = samples[: n * win].reshape(n, win)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    peak = np.max(rms) + 1e-12
    db = 20 * np.log10(rms / peak)
    active = np.where(db > silence_db)[0]
    if active.size == 0:
        return 0.0
    interior = db[active[0]: active[-1] + 1]
    return float((interior <= silence_db).mean())
