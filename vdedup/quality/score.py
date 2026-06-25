"""Stage 8 — composite quality scalar Q and the hard TERRIBLE gate.

Video and audio composites are computed separately then combined:
    Q = Q_video + lambda * Q_audio
A file is `terrible` if it trips any absolute floor in EITHER modality (but the
audio sub-gate never flags a silent/video-only file — absence of audio is not
bad audio). Absolute component values are retained for the gate; the decision
engine re-normalises Q within a cluster for *ranking*.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import video_nr, audio_nr


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class QualityResult:
    components: dict = field(default_factory=dict)
    Q_video: float = 0.0
    Q_audio: float = 0.0
    Q_composite: float = 0.0
    terrible: bool = False
    terrible_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(self.components)
        return d


def score_file(meta: dict, gray_native: np.ndarray, audio: np.ndarray, sr: int,
               has_audio: bool, cfg, dover_tech: float | None = None) -> QualityResult:
    q = cfg.quality
    w = meta.get("declared_w") or (gray_native.shape[2] if gray_native.ndim == 3 else 0)
    h = meta.get("declared_h") or (gray_native.shape[1] if gray_native.ndim == 3 else 0)

    # ---- video components ----
    reff, delta = video_nr.r_eff(w, h, gray_native, q.reff_detail_band, q.reff_detail_ref)
    bpp = video_nr.bpp_norm(meta.get("declared_bitrate"), w, h, meta.get("declared_fps"),
                            meta.get("vcodec"), q.codec_efficiency)
    block = video_nr.blockiness(gray_native)     # stored diagnostics (see w_artifact note)
    band = video_nr.banding(gray_native)
    artifact = _clip01(block / 2.0)
    if dover_tech is None:                        # DSP technical-quality proxy (detail-based)
        dover_tech = _clip01(delta / q.reff_detail_ref)

    n_reff = _clip01(reff / (1920 * 1080))
    n_bpp = _clip01((bpp or 0.0) / 0.15)
    Q_video = (q.w_dover * dover_tech + q.w_reff * n_reff + q.w_bpp * n_bpp
               - q.w_artifact * artifact)

    # ---- audio components ----
    tracks = meta.get("audio_tracks") or []
    a0 = tracks[0] if tracks else {}
    channels = a0.get("channels") or 0
    abr = audio_nr.abr_norm(a0.get("audio_bitrate"), channels, a0.get("acodec"), q.codec_efficiency)
    if has_audio and audio.size > 0:
        bw = audio_nr.audio_bandwidth(audio, sr)
        clip = audio_nr.clip_ratio(audio)
        drop = audio_nr.dropout_ratio(audio, sr)
    else:
        bw = clip = drop = 0.0

    n_bw = _clip01(bw / (sr / 2))
    n_abr = _clip01((abr or 0.0) / 96000.0)
    n_ch = _clip01(channels / 2.0)
    Q_audio = (q.w_bw * n_bw + q.w_abr * n_abr + q.w_channels * n_ch
               - q.w_clip * _clip01(clip + drop))
    if not has_audio:
        Q_audio = 0.0

    Q_composite = Q_video + q.lam * Q_audio

    # ---- TERRIBLE gate ----
    reasons = []
    if reff < q.terrible_reff_px:
        reasons.append(f"R_eff<{q.terrible_reff_px:.0f}px({reff:.0f})")
    if dover_tech < q.terrible_dover:
        reasons.append(f"dover<{q.terrible_dover}")
    if bpp is not None and bpp < q.terrible_bpp_norm:
        reasons.append(f"bpp<{q.terrible_bpp_norm}")
    if has_audio and audio.size > 0:
        if 0 < bw < q.terrible_audio_bw_hz:
            reasons.append(f"audio_bw<{q.terrible_audio_bw_hz:.0f}Hz({bw:.0f})")
        if clip > q.terrible_clip_ratio:
            reasons.append("clipping")
        if drop > q.terrible_dropout_ratio:
            reasons.append("dropouts")

    components = dict(R_eff=reff, delta=delta, bpp_norm=bpp, dover_tech=dover_tech,
                      blockiness=block, banding=band, artifact=artifact,
                      audio_bw_hz=bw, abr_norm=abr, channels=channels,
                      clip_ratio=clip, dropout_ratio=drop)
    return QualityResult(components=components, Q_video=float(Q_video), Q_audio=float(Q_audio),
                         Q_composite=float(Q_composite), terrible=bool(reasons),
                         terrible_reason=";".join(reasons))
