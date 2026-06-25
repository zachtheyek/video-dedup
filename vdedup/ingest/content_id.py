"""Content identity = hash of a normalized decoded keyframe stream.

A pure file hash misses container re-muxes (same video, different MP4/MKV
wrapper) and trivial metadata edits, which would look like distinct files.
Hashing a deterministic decode of a fixed set of evenly-spaced frames at a fixed
small size (post-deletterbox) gives an exact-content identity that survives
remuxing while still distinguishing genuine re-encodes.

(The design names xxh3-128; we use stdlib BLAKE2b-128 to avoid a non-stdlib
dependency. Identical role: fast, collision-resistant content fingerprint.)

Deviation from the design: we hash the *full* frame, not the deletterboxed one.
The design specified post-deletterbox so a letterbox-variant remux would still
match — but adding/removing black bars requires *re-encoding* (you cannot `-c
copy` bars in), so such a file always has distinct pixels and a distinct id
regardless. Hashing the full frame removes a dependence on the (slightly
sampling-sensitive) crop detector and makes the id exactly remux-invariant.
"""
from __future__ import annotations

import hashlib

from ..media import ffmpeg

N_FRAMES = 16
HASH_EDGE = 64


def content_id(path, width: int | None, height: int | None, duration: float | None) -> str:
    """Deterministic, remux-invariant content id. Falls back to a file-stat hash
    for files that cannot be decoded as video (e.g., audio-only inputs)."""
    if not width or not height or not duration or duration <= 0:
        return _stat_fallback(path)
    # Quantize duration to a 0.5s grid so container-specific duration jitter
    # (e.g. MP4 30.000 vs MKV 30.023 for a -c copy remux) does not shift which
    # frames get sampled, while genuinely different lengths still map elsewhere.
    dur_q = max(0.5, round(duration / 0.5) * 0.5)
    try:
        sample_fps = max(0.01, N_FRAMES / dur_q)
        frames, _ = ffmpeg.decode_frames(path, sample_fps, HASH_EDGE, HASH_EDGE, gray=True)
    except ffmpeg.FFmpegError:
        return _stat_fallback(path)
    if frames.shape[0] == 0:
        return _stat_fallback(path)
    frames = frames[:N_FRAMES]
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{frames.shape[0]}x{HASH_EDGE}".encode())
    h.update(frames.tobytes())
    return h.hexdigest()


def _stat_fallback(path) -> str:
    import os
    st = os.stat(path)
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{path}:{st.st_size}".encode())
    return "stat-" + h.hexdigest()
