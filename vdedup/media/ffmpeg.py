"""Thin ffmpeg/ffprobe wrappers.

Temporal sampling uses the `fps` filter, which resamples on decoder PTS (so it is
correct for VFR sources): output frame k corresponds to content time k / fps.
That is the design's "PTS, never frame_index/fps" requirement, satisfied by
letting ffmpeg do the PTS-aware resampling.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class FFmpegError(RuntimeError):
    pass


def _run(cmd: list[str]) -> bytes:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd[:6])} ... -> rc={p.returncode}\n{p.stderr.decode()[-800:]}")
    return p.stdout


def probe(path: str | Path) -> dict:
    out = _run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    return json.loads(out)


def fit_long_side(w: int, h: int, long: int | None) -> tuple[int, int]:
    if long is None or max(w, h) <= long:
        return (w - w % 2, h - h % 2)
    if w >= h:
        nw = long
        nh = max(2, round(h * long / w))
    else:
        nh = long
        nw = max(2, round(w * long / h))
    return (nw - nw % 2, nh - nh % 2)


def _vf(crop: tuple[int, int, int, int] | None, out_w: int, out_h: int,
        fps: float, gray: bool) -> str:
    # Normalize the PTS origin first so container start_time differences (e.g. a
    # -c copy remux that sets start_time=0.023) do not shift which frames the
    # fps sampler picks. Also gives every file a consistent t=0 content origin.
    parts = ["setpts=PTS-STARTPTS"]
    if crop is not None:
        cw, ch, cx, cy = crop
        parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
    parts.append(f"fps={fps}")
    parts.append(f"scale={out_w}:{out_h}:flags=bicubic")
    parts.append("format=gray" if gray else "format=rgb24")
    return ",".join(parts)


def decode_frames(path: str | Path, fps: float, out_w: int, out_h: int, *,
                  crop: tuple[int, int, int, int] | None = None,
                  gray: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (frames, times). frames is [n,h,w] (gray) or [n,h,w,3] (rgb) uint8;
    times[k] = k / fps seconds."""
    ch = 1 if gray else 3
    vf = _vf(crop, out_w, out_h, fps, gray)
    raw = _run([FFMPEG, "-v", "error", "-i", str(path), "-vf", vf,
                "-pix_fmt", "gray" if gray else "rgb24", "-f", "rawvideo", "-"])
    frame_bytes = out_w * out_h * ch
    n = len(raw) // frame_bytes
    arr = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8)
    shape = (n, out_h, out_w) if gray else (n, out_h, out_w, 3)
    frames = arr.reshape(shape)
    times = np.arange(n, dtype=np.float64) / fps
    return frames, times


def decode_audio(path: str | Path, sample_rate: int, *, track: int = 0,
                 loudnorm: bool = False) -> np.ndarray:
    """Decode one audio track to mono float32 at `sample_rate`. Returns [] if the
    track has no audio."""
    af = ["asetpts=PTS-STARTPTS", "aresample=" + str(sample_rate)]
    if loudnorm:
        af.insert(1, "loudnorm=I=-23:LRA=7:tp=-2")
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-map", f"0:a:{track}?",
           "-ac", "1", "-ar", str(sample_rate), "-af", ",".join(af),
           "-f", "f32le", "-"]
    try:
        raw = _run(cmd)
    except FFmpegError:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32).copy()


def has_audio(probe_json: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe_json.get("streams", []))
