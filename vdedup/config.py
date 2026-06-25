"""Central configuration.

Every tunable from the design document's "Open questions and tunables"
(Section 20) lives here with a documented default. Load order:
defaults -> optional YAML file -> explicit overrides. Nothing in the pipeline
reads magic numbers directly; they all come from a `Config` instance so a run is
fully reproducible from one serialisable object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml


@dataclass
class VisionCfg:
    sample_fps: float = 2.0          # decoded frames per wall-clock second (PTS-based)
    embed_size: int = 288            # SSCD default input edge
    phash_size: int = 64            # bits in the perceptual hash
    entropy_min: float = 2.0         # luminance-histogram Shannon entropy floor (bits): rejects
                                     # black/fades/solid cards (~0-1.5) while keeping real content
    edge_density_min: float = 0.01   # secondary informativeness gate
    cropdetect_frames: int = 60      # frames sampled for deletterbox max-projection
    use_sscd: bool = True            # use SSCD embedding when weights are available
    embed_dim: int = 512             # SSCD output dim (pre-PCA)
    flip_augment: bool = False       # index horizontal-flip descriptors too (mirror re-uploads)


@dataclass
class AudioCfg:
    sample_rate: int = 16000         # mono resample target
    frame_size: int = 4096           # STFT window
    hop_size: int = 2048             # STFT hop
    peak_neighborhood: int = 20      # local-max neighbourhood (bins) for constellation peaks
    fan_value: int = 15              # peaks paired with each anchor
    min_hash_time_delta: float = 0.0
    max_hash_time_delta: float = 5.0  # seconds; pair window for landmark hashing
    activity_flux_pct: float = 25.0   # spectral-flux percentile below which frames are inactive
    activity_energy_pct: float = 20.0  # energy percentile floor
    loudnorm: bool = True


@dataclass
class CandidateCfg:
    knn_k: int = 10                  # visual ANN neighbours per query frame
    visual_sim_min: float = 0.6      # cosine floor for a visual near-neighbour to count
    min_vote_mass: float = 3.0       # IDF-weighted vote needed to surface a candidate pair
    min_shared_audio_hashes: int = 5
    idf_smoothing: float = 1.0
    use_faiss: bool = False          # see VisualIndex: faiss+torch deadlock on macOS; numpy default


@dataclass
class AlignCfg:
    offset_bin: float = 0.25         # seconds; Hough histogram bin (audio supports finer than 2fps)
    alpha_grid: tuple[float, float, float] = (0.96, 1.04, 0.005)  # coarse scale search (lo, hi, step)
    inlier_tol: float = 0.5          # seconds; residual tolerance for inlier membership
    min_inliers: int = 6
    min_span_seconds: float = 3.0    # matched evidence must span >= T sec of A's timeline
    max_residual_std: float = 0.4    # seconds
    audio_offset_var: float = 0.05   # prior offset variance for an audio inlier -> edge weight
    visual_offset_var: float = 0.30  # prior offset variance for a vision inlier
    piecewise_min_segment: float = 5.0  # min seconds per piece before declaring a breakpoint
    scale_search: bool = False       # coarse alpha grid search (PAL/NTSC speed-up); off = alpha fixed at 1
    cycle_residual_tol: float = 1.5  # drop edges whose timeline residual exceeds this (seconds)


@dataclass
class QualityCfg:
    # video composite weights
    w_dover: float = 0.45
    w_reff: float = 0.35
    w_bpp: float = 0.20
    # blockiness/banding are unreliable proxies without real footage (synthetic
    # flat regions read as banding); computed + stored as diagnostics but
    # zero-weighted by default. Raise on a calibrated library, or rely on DOVER.
    w_artifact: float = 0.0
    # audio composite weights
    w_bw: float = 0.5
    w_abr: float = 0.3
    w_channels: float = 0.2
    w_clip: float = 0.3
    lam: float = 0.25                # audio weight in fused Q = Q_video + lam*Q_audio
    reff_detail_band: tuple[float, float] = (0.25, 0.9)  # normalised radial freq band for delta
    reff_detail_ref: float = 0.0015  # delta of "full-detail" content; R_eff = pixels*clip(delta/ref)
    # TERRIBLE gate absolute floors (calibrate to library; these are conservative defaults)
    terrible_reff_px: float = 640 * 360 * 0.40   # effective-pixels floor (~ sub-360p detail)
    terrible_dover: float = 0.10                  # absolute NR technical floor [0,1] (conservative)
    terrible_bpp_norm: float = 0.015              # bits/pixel/frame floor (H.264-normalised)
    terrible_audio_bw_hz: float = 8000.0          # spectral rolloff floor (heavy transcode)
    terrible_clip_ratio: float = 0.02             # fraction of full-scale samples
    terrible_dropout_ratio: float = 0.05
    codec_efficiency: dict[str, float] = field(default_factory=lambda: {
        # bitrate divisor relative to H.264 reference (higher = more efficient codec)
        "h264": 1.0, "hevc": 2.0, "h265": 2.0, "av1": 2.4, "vp9": 1.8, "mpeg4": 0.7,
        "aac": 1.0, "opus": 1.3, "mp3": 0.7, "vorbis": 1.1, "flac": 1.0, "ac3": 0.8,
    })
    use_dover: bool = True
    use_vmaf: bool = True            # alignment-relative full-reference cross-check (native ffmpeg)


@dataclass
class DecideCfg:
    eps_full_length: float = 0.05    # is_full_length covers >= (1-eps) of canonical span
    contain_tol: float = 0.5         # seconds tolerance for interval containment
    rule6_quality_margin: float | None = None  # Δ: keep an absorbed clip if Q exceeds full's by > Δ (None=off)
    set_cover_pass: bool = False     # optional redundancy pass; off = continuity preferred
    audio_variant_keep_both: bool = True  # keep both on distinct-language redub; dominate on same-language
    use_lid: bool = False            # language-ID pass for untagged audio_variant pairs


@dataclass
class ActionCfg:
    mode: str = "dry-run"            # dry-run | apply
    quarantine_dir: str = "quarantine"
    quarantine_ttl_days: int = 30
    auto_approve_confidence: float | None = None  # None = always require interactive approval
    thin_margin: float = 0.05        # dominator margin below which a prune routes to review


@dataclass
class Config:
    root: str = "."                  # library root to scan
    data_dir: str = "data"           # catalog db + indexes + caches live here
    models_dir: str = "models"       # downloaded model weights (persist across runs)
    workers: int = 0                 # 0 = os.cpu_count()
    device: str = "auto"             # auto | cpu | mps | cuda  (for torch extractors)
    vision: VisionCfg = field(default_factory=VisionCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    candidate: CandidateCfg = field(default_factory=CandidateCfg)
    align: AlignCfg = field(default_factory=AlignCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    decide: DecideCfg = field(default_factory=DecideCfg)
    action: ActionCfg = field(default_factory=ActionCfg)

    # ---- IO helpers -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        cfg = cls()
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            cfg = cls.from_dict(raw)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        sub = {
            "vision": VisionCfg, "audio": AudioCfg, "candidate": CandidateCfg,
            "align": AlignCfg, "quality": QualityCfg, "decide": DecideCfg,
            "action": ActionCfg,
        }
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k in sub and isinstance(v, dict):
                kwargs[k] = sub[k](**v)
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)
