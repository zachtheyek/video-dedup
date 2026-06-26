"""Stage 5b — acceptance gate + modality-disagreement decision table.

The governing principle (design Section 8): *cluster membership tracks the
video*. Audio sharpens precision and robustness but a shared soundtrack alone is
too weak to call two files the same title for pruning.

| Vision   | Audio    | Interpretation                          | Action                         |
|----------|----------|-----------------------------------------|--------------------------------|
| agrees   | agrees   | same content, intact soundtrack         | accept `both`, highest conf    |
| agrees   | absent   | one/both silent or video-only           | accept `visual`, no penalty    |
| agrees   | disagrees| same video, different soundtrack        | accept `audio_variant` on vision|
| disagrees| agrees   | same audio over different video         | route to review (do not merge) |
| disagrees| absent   | coincidental                            | reject                         |
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .offset import MatchPairs, AlignFit, fit_offset

MODALITY_TABLE = ("both", "visual", "audio_variant", "audio", "reject")


@dataclass
class EdgeDecision:
    accept: bool                 # becomes a (video-grounded) cluster edge
    route_review: bool
    modality: str                # one of MODALITY_TABLE
    alpha: float
    beta: float
    v_inliers: int
    a_inliers: int
    span_seconds: float
    residual_std: float
    audio_agrees: bool
    confidence: float
    piecewise: bool
    reason: str


def _passes_gate(fit: AlignFit, cfg) -> bool:
    return (fit.n_inliers >= cfg.min_inliers
            and fit.span_seconds >= cfg.min_span_seconds
            and fit.residual_std <= cfg.max_residual_std)


def _fit(pairs: MatchPairs, cfg, alpha_grid) -> AlignFit:
    return fit_offset(pairs, offset_bin=cfg.offset_bin, alpha_grid=alpha_grid,
                      inlier_tol=cfg.inlier_tol,
                      piecewise_min_segment=cfg.piecewise_min_segment)


def _confidence(n_inliers: int, span: float, rstd: float, cross_modal: bool) -> float:
    """Monotone in inlier count, span, inverse residual, and cross-modal agreement."""
    base = (1.0 - np.exp(-n_inliers / 8.0)) * (1.0 - np.exp(-span / 10.0))
    sharp = 1.0 / (1.0 + rstd)
    bonus = 1.15 if cross_modal else 1.0
    return float(min(1.0, base * sharp * bonus))


def verify_pair(pairs: MatchPairs, *, has_audio_a: bool, has_audio_b: bool, cfg,
                alpha_grid=(1.0,)) -> EdgeDecision:
    fused = _fit(pairs, cfg, alpha_grid)

    vis = pairs.subset(~pairs.is_audio)
    aud = pairs.subset(pairs.is_audio)
    vfit = _fit(vis, cfg, alpha_grid)
    afit = _fit(aud, cfg, alpha_grid)

    vstate_ok = _passes_gate(vfit, cfg)
    astate_ok = _passes_gate(afit, cfg)
    both_have_audio = has_audio_a and has_audio_b
    piecewise = bool(fused.piecewise) or bool(vfit.piecewise)

    def decide(accept, review, modality, fit, audio_agrees, reason, cross_modal=False):
        return EdgeDecision(
            accept=accept, route_review=review or piecewise, modality=modality,
            alpha=fit.alpha, beta=fit.beta, v_inliers=vfit.n_inliers,
            a_inliers=afit.n_inliers, span_seconds=fit.span_seconds,
            residual_std=fit.residual_std, audio_agrees=audio_agrees,
            confidence=_confidence(fit.n_inliers, fit.span_seconds, fit.residual_std, cross_modal),
            piecewise=piecewise, reason=reason)

    if vstate_ok and astate_ok:
        av = vfit.beta - afit.beta
        if abs(av) <= cfg.inlier_tol:
            return decide(True, False, "both", fused, True,
                          "vision+audio agree on offset", cross_modal=True)
        # Both internally consistent but on different offsets. A small, consistent
        # gap is an A/V desync within one file (Section 17); a large one means the
        # soundtrack is genuinely different content. Either way trust the video.
        if abs(av) <= cfg.av_desync_max:
            return decide(True, True, "audio_variant", vfit, False,
                          f"A/V desync ~{av:+.2f}s (audio vs video); trusting video")
        return decide(True, True, "audio_variant", vfit, False,
                      "vision and audio give different offsets; trusting video")

    if vstate_ok and not astate_ok:
        if both_have_audio and afit.n_inliers < cfg.min_inliers:
            # video aligns over a real span, yet two audio-bearing files share
            # almost no audio anchors -> replaced/muted soundtrack (redub etc.)
            return decide(True, False, "audio_variant", vfit, False,
                          "video aligns but soundtracks differ (replaced/muted audio)")
        return decide(True, False, "visual", vfit, False,
                      "video aligns; audio absent or silent (no penalty)")

    if not vstate_ok and astate_ok:
        # same audio over different video: commentary / reaction / static image.
        return decide(False, True, "audio", afit, True,
                      "audio aligns but video does not; not auto-merged")

    return decide(False, False, "reject", fused, False, "no consistent shared sequence")
