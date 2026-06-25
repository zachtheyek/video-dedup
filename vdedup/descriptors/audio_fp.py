"""Stage 3 (audio) — landmark / constellation fingerprints (Shazam/Dejavu-style).

Take the log-magnitude spectrogram, pick robust local spectral peaks, and hash
*pairs* of peaks into (f_anchor, f_target, dt) tokens, each carrying its anchor
timestamp. Exact-hash-lookupable (matching is an inverted-index probe, not a
nearest-neighbour search), invariant to every transform that defeats vision
(resolution, codec, crop, letterbox, flip, overlay), and — the architectural
payoff — a confirmed audio match yields the same {(t_A, t_B)} matched-pair
structure as a visual match, so both feed one offset estimator.

An energy/spectral-flux activity filter suppresses peaks from silence/room-tone
so silence cannot bridge unrelated files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter


@dataclass
class AudioFingerprint:
    hashes: np.ndarray   # uint64 landmark hashes
    times: np.ndarray    # float anchor times (seconds)

    def __len__(self) -> int:
        return int(self.hashes.shape[0])


def _spectrogram(samples: np.ndarray, sr: int, frame_size: int, hop: int):
    f, t, Z = signal.stft(samples, fs=sr, nperseg=frame_size,
                          noverlap=frame_size - hop, boundary=None, padded=False)
    S = np.abs(Z)
    logS = 20 * np.log10(S + 1e-6)
    return f, t, logS


def _activity(logS: np.ndarray, energy_pct: float, flux_pct: float) -> np.ndarray:
    """Per-frame mask: True where the frame is energetic AND spectrally varying."""
    energy = logS.mean(axis=0)
    flux = np.empty_like(energy)
    flux[0] = 0.0
    diff = np.diff(logS, axis=1)
    flux[1:] = np.maximum(diff, 0).sum(axis=0)
    e_thr = np.percentile(energy, energy_pct)
    f_thr = np.percentile(flux, flux_pct)
    return (energy >= e_thr) | (flux >= f_thr)


def _peaks(logS: np.ndarray, neighborhood: int, active: np.ndarray) -> list[tuple[int, int]]:
    footprint = np.ones((neighborhood, neighborhood), dtype=bool)
    local_max = maximum_filter(logS, footprint=footprint) == logS
    amp_thr = logS.mean() + 0.5 * logS.std()
    strong = logS > amp_thr
    peaks = local_max & strong
    peaks[:, ~active] = False                      # drop peaks in inactive frames
    fi, ti = np.where(peaks)
    order = np.argsort(ti)
    return list(zip(ti[order], fi[order]))         # (time_idx, freq_idx) sorted by time


def fingerprint(samples: np.ndarray, sr: int, *, frame_size: int = 4096, hop: int = 2048,
                peak_neighborhood: int = 20, fan_value: int = 15,
                min_dt: float = 0.0, max_dt: float = 5.0,
                energy_pct: float = 20.0, flux_pct: float = 25.0) -> AudioFingerprint:
    if samples.size < frame_size:
        return AudioFingerprint(np.zeros(0, dtype=np.uint64), np.zeros(0))

    f, t, logS = _spectrogram(samples, sr, frame_size, hop)
    active = _activity(logS, energy_pct, flux_pct)
    peaks = _peaks(logS, peak_neighborhood, active)
    frame_dt = hop / sr

    hashes: list[int] = []
    times: list[float] = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        for j in range(1, fan_value + 1):
            if i + j >= n:
                break
            t2, f2 = peaks[i + j]
            dt = (t2 - t1) * frame_dt
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            dt_q = int(round(dt / frame_dt))
            h = ((int(f1) & 0x3FF) << 20) | ((int(f2) & 0x3FF) << 10) | (dt_q & 0x3FF)
            hashes.append(h)
            times.append(float(t1 * frame_dt))
    return AudioFingerprint(np.asarray(hashes, dtype=np.uint64), np.asarray(times, dtype=np.float64))


def matched_pairs(fp_a: AudioFingerprint, fp_b: AudioFingerprint):
    """Return (t_a, t_b) arrays for every hash collision between two fingerprints.
    A shared hash with anchor times (ta, tb) is one matched timestamp pair."""
    if len(fp_a) == 0 or len(fp_b) == 0:
        return np.zeros(0), np.zeros(0)
    from collections import defaultdict
    idx: dict[int, list[float]] = defaultdict(list)
    for h, tb in zip(fp_b.hashes.tolist(), fp_b.times.tolist()):
        idx[h].append(tb)
    ta_out, tb_out = [], []
    for h, ta in zip(fp_a.hashes.tolist(), fp_a.times.tolist()):
        if h in idx:
            for tb in idx[h]:
                ta_out.append(ta)
                tb_out.append(tb)
    return np.asarray(ta_out), np.asarray(tb_out)
