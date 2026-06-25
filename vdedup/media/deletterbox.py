"""Stage 2 — deletterboxing / depillarboxing via robust max-projection.

Bars are rows/columns that are black across *every* frame, so the per-pixel
*maximum* luminance over a sample of frames stays dark there. A per-frame
approach is fooled by dark scenes; the max-projection is not. Returns the active
picture rectangle as an ffmpeg crop spec (w, h, x, y), or None when the whole
frame is active.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ffmpeg


@dataclass(frozen=True)
class CropRect:
    w: int
    h: int
    x: int
    y: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.w, self.h, self.x, self.y)


def detect_crop(path, width: int, height: int, duration: float, *,
                n_frames: int = 60, black_thresh: int = 24,
                min_bar_frac: float = 0.02) -> CropRect | None:
    ow, oh = ffmpeg.fit_long_side(width, height, None)
    sample_fps = max(0.05, min(4.0, n_frames / max(duration, 1.0)))
    frames, _ = ffmpeg.decode_frames(path, sample_fps, ow, oh, gray=True)
    if frames.shape[0] == 0:
        return None
    frames = frames[:n_frames]
    proj = frames.max(axis=0)                      # [h, w] per-pixel max luminance
    row_active = proj.max(axis=1) > black_thresh   # rows with any signal
    col_active = proj.max(axis=0) > black_thresh
    if not row_active.any() or not col_active.any():
        return None                                 # all-black sample; don't crop

    ys = np.where(row_active)[0]
    xs = np.where(col_active)[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1

    # ignore negligible bars (< min_bar_frac of the dimension)
    if (y0 < oh * min_bar_frac and oh - y1 < oh * min_bar_frac and
            x0 < ow * min_bar_frac and ow - x1 < ow * min_bar_frac):
        return None

    cw = (x1 - x0); ch = (y1 - y0)
    cw -= cw % 2; ch -= ch % 2
    if cw <= 0 or ch <= 0 or (cw == ow and ch == oh):
        return None
    return CropRect(cw, ch, x0, y0)
