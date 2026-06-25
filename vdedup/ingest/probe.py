"""Parse ffprobe JSON into the declared stream/audio metadata the catalog stores.

Everything here is the container's *claim*; it is a prior and sanity check, never
ground truth. Effective resolution, true extent and quality are measured later
from decoded signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _frac(s: str | None) -> float | None:
    if not s or s == "0/0":
        return None
    try:
        num, den = s.split("/")
        den = float(den)
        return float(num) / den if den else None
    except (ValueError, ZeroDivisionError):
        try:
            return float(s)
        except ValueError:
            return None


@dataclass
class StreamInfo:
    has_video: bool
    width: int | None
    height: int | None
    fps: float | None
    vcodec: str | None
    pix_fmt: str | None
    bitrate: int | None
    duration: float | None
    vfr: bool
    has_audio: bool
    audio_tracks: list[dict] = field(default_factory=list)
    default_track: int | None = None

    def stream_meta(self, active_crop=None) -> dict:
        return {
            "declared_w": self.width, "declared_h": self.height, "declared_fps": self.fps,
            "declared_bitrate": self.bitrate, "vcodec": self.vcodec, "pix_fmt": self.pix_fmt,
            "declared_duration": self.duration, "active_crop": active_crop, "vfr": self.vfr,
        }


def parse_probe(probe: dict) -> StreamInfo:
    streams = probe.get("streams", [])
    fmt = probe.get("format", {})
    vstreams = [s for s in streams if s.get("codec_type") == "video"
                and s.get("disposition", {}).get("attached_pic", 0) == 0]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]

    dur = None
    for src in (fmt.get("duration"), *(v.get("duration") for v in vstreams)):
        if src:
            try:
                dur = float(src)
                break
            except ValueError:
                pass

    if vstreams:
        v = vstreams[0]
        avg = _frac(v.get("avg_frame_rate"))
        r = _frac(v.get("r_frame_rate"))
        fps = avg or r
        # VFR heuristic: average and base frame rates materially disagree
        vfr = bool(avg and r and abs(avg - r) / max(r, 1e-6) > 0.01)
        si_video = dict(
            has_video=True, width=v.get("width"), height=v.get("height"), fps=fps,
            vcodec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"),
            bitrate=int(v["bit_rate"]) if v.get("bit_rate", "").isdigit() else None,
            duration=dur, vfr=vfr)
    else:
        si_video = dict(has_video=False, width=None, height=None, fps=None, vcodec=None,
                        pix_fmt=None, bitrate=None, duration=dur, vfr=False)

    tracks = []
    default_track = None
    for i, a in enumerate(astreams):
        if a.get("disposition", {}).get("default", 0) == 1 and default_track is None:
            default_track = i
        tracks.append({
            "track": i,
            "acodec": a.get("codec_name"),
            "sample_rate": int(a["sample_rate"]) if a.get("sample_rate") else None,
            "channels": a.get("channels"),
            "audio_bitrate": int(a["bit_rate"]) if a.get("bit_rate", "").isdigit() else None,
            "lang_tag": (a.get("tags", {}) or {}).get("language"),
            "lang_detected": None,
        })
    if astreams and default_track is None:
        default_track = 0

    return StreamInfo(has_audio=bool(astreams), audio_tracks=tracks,
                      default_track=default_track, **si_video)
