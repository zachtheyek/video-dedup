"""Stage 5a — fused robust offset estimation (early fusion).

Both modalities produce the same evidence: matched timestamp pairs {(t_A, t_B)}
under  t_B = alpha * t_A + beta,  alpha ~= 1. We pool them into ONE weighted
offset vote (each pair weighted by modality-reliability prior x IDF mass), take
the histogram peak as beta_0, collect inliers, then refine (alpha, beta) by
weighted least squares on the inliers. Audio anchors are dense and sub-second so
they dominate beta's precision; vision fills in where audio is silent/replaced.

A second pass on the residual pairs detects a single breakpoint (piecewise
alignment) — the director's-cut / inserted-scene case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchPairs:
    """Matched timestamp pairs for one candidate file pair (A, B)."""
    ta: np.ndarray          # anchor/frame times in A (seconds)
    tb: np.ndarray          # corresponding times in B (seconds)
    weight: np.ndarray      # per-pair weight (modality prior x IDF mass)
    is_audio: np.ndarray    # bool mask: True = audio landmark, False = visual kNN

    @classmethod
    def empty(cls) -> "MatchPairs":
        z = np.zeros(0)
        return cls(z, z.copy(), z.copy(), np.zeros(0, dtype=bool))

    def __len__(self) -> int:
        return int(self.ta.shape[0])

    def subset(self, mask: np.ndarray) -> "MatchPairs":
        return MatchPairs(self.ta[mask], self.tb[mask], self.weight[mask], self.is_audio[mask])


@dataclass
class AlignFit:
    alpha: float
    beta: float
    n_inliers: int
    v_inliers: int          # visual inliers
    a_inliers: int          # audio inliers
    span_seconds: float     # extent of A covered by inliers
    residual_std: float
    peak_mass: float        # weighted vote mass in the peak
    inlier_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    audio_agrees: bool = False   # do audio inliers corroborate the fused offset?
    piecewise: list[tuple[float, float, float]] | None = None  # [(alpha,beta,span), ...]


def _weighted_peak(betas: np.ndarray, weights: np.ndarray, bin_w: float) -> tuple[float, float]:
    """Return (peak_center_beta, peak_mass) of a weighted 1-D histogram."""
    if betas.size == 0:
        return 0.0, 0.0
    lo, hi = betas.min(), betas.max()
    if hi - lo < bin_w:
        return float(np.average(betas, weights=weights)), float(weights.sum())
    nbins = int(np.ceil((hi - lo) / bin_w)) + 1
    edges = lo + bin_w * np.arange(nbins + 1)
    hist, _ = np.histogram(betas, bins=edges, weights=weights)
    k = int(np.argmax(hist))
    center = 0.5 * (edges[k] + edges[k + 1])
    return float(center), float(hist[k])


def _fit_single(pairs: MatchPairs, alpha_grid, bin_w, tol) -> tuple[float, float, np.ndarray, float]:
    """Best (alpha, beta, inlier_mask, peak_mass) over a coarse alpha grid."""
    best = (1.0, 0.0, np.zeros(len(pairs), dtype=bool), -1.0)
    for alpha in alpha_grid:
        betas = pairs.tb - alpha * pairs.ta
        beta0, mass = _weighted_peak(betas, pairs.weight, bin_w)
        inliers = np.abs(betas - beta0) <= tol
        if mass > best[3]:
            best = (alpha, beta0, inliers, mass)
    return best


def fit_offset(pairs: MatchPairs, *, offset_bin: float = 0.25,
               alpha_grid=(1.0,), inlier_tol: float = 0.5,
               piecewise_min_segment: float = 5.0,
               detect_piecewise: bool = True) -> AlignFit:
    if len(pairs) == 0:
        return AlignFit(1.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0)

    alpha, beta0, inliers, mass = _fit_single(pairs, alpha_grid, offset_bin, inlier_tol)

    # weighted least-squares refine of (alpha, beta) on inliers
    if inliers.sum() >= 2 and np.ptp(pairs.ta[inliers]) > 1e-6:
        ta_in, tb_in, w_in = pairs.ta[inliers], pairs.tb[inliers], pairs.weight[inliers]
        W = np.sqrt(w_in)
        A = np.stack([ta_in * W, W], axis=1)
        sol, *_ = np.linalg.lstsq(A, tb_in * W, rcond=None)
        alpha, beta = float(sol[0]), float(sol[1])
        resid = pairs.tb - (alpha * pairs.ta + beta)
        inliers = np.abs(resid) <= inlier_tol
    else:
        beta = beta0
        resid = pairs.tb - (alpha * pairs.ta + beta)

    inl = inliers
    n = int(inl.sum())
    v_in = int((inl & ~pairs.is_audio).sum())
    a_in = int((inl & pairs.is_audio).sum())
    span = float(np.ptp(pairs.ta[inl])) if n else 0.0
    rstd = float(np.std(resid[inl])) if n else 0.0
    audio_agrees = a_in > 0 and (np.abs(resid[inl & pairs.is_audio]) <= inlier_tol).all()

    piecewise = None
    if detect_piecewise:
        piecewise = _detect_piecewise(pairs, inl, offset_bin, inlier_tol,
                                      piecewise_min_segment, alpha, beta, span)

    return AlignFit(alpha, beta, n, v_in, a_in, span, rstd, mass,
                    inlier_mask=inl, audio_agrees=audio_agrees, piecewise=piecewise)


def _detect_piecewise(pairs, primary_inl, bin_w, tol, min_seg, alpha1, beta1, span1):
    """Detect a single second consistent offset among the residual pairs.

    Conservative: the second segment must itself be a substantial, well-spanned,
    clearly-separated offset, not the scatter that any kNN match leaves behind —
    otherwise clean full-vs-full matches get false 'different cut' flags.
    """
    n_primary = int(primary_inl.sum())
    out = ~primary_inl
    # require a real residual population, not a handful of kNN outliers
    if out.sum() < max(10, 0.5 * n_primary):
        return None
    rest = pairs.subset(out)
    a2, b2, inl2, mass2 = _fit_single(rest, (alpha1,), bin_w, tol)
    n2 = int(inl2.sum())
    if n2 < max(8, 0.4 * n_primary):
        return None
    span2 = float(np.ptp(rest.ta[inl2])) if inl2.any() else 0.0
    if span2 < min_seg or span1 < min_seg:
        return None
    if abs(b2 - beta1) <= 2 * tol:        # the two offsets must be genuinely different
        return None
    return [(alpha1, beta1, span1), (a2, b2, span2)]
