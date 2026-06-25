"""Stage 8 — no-reference VQA (DOVER) behind a swappable interface.

DOVER's *technical* sub-score targets the prune axis (compression/blur/blocking).
We wire it via pyiqa when available; otherwise the caller uses the DSP technical
proxy in score.py (R_eff detail + inverse artifacts), so the gate always works.
"""
from __future__ import annotations

import numpy as np


class DoverVQA:
    def __init__(self, device: str = "auto"):
        self._metric = None
        self._failed = False
        self.device = device

    def _load(self):
        if self._metric is not None or self._failed:
            return
        try:
            import torch
            import pyiqa
            dev = self.device
            if dev == "auto":
                dev = "mps" if torch.backends.mps.is_available() else (
                    "cuda" if torch.cuda.is_available() else "cpu")
            # DOVER weights can be large; tolerate any failure.
            self._metric = pyiqa.create_metric("dover", device=dev)
        except Exception:
            self._failed = True

    @property
    def available(self) -> bool:
        self._load()
        return self._metric is not None

    def technical(self, frames_rgb: np.ndarray) -> float | None:
        """Return DOVER technical score normalised to ~[0,1], or None."""
        if not self.available or frames_rgb.shape[0] == 0:
            return None
        try:
            import torch
            # DOVER expects a video tensor [1, C, T, H, W] in [0,1]
            x = torch.from_numpy(np.ascontiguousarray(frames_rgb)).float() / 255.0
            x = x.permute(3, 0, 1, 2).unsqueeze(0)  # -> [1,C,T,H,W]
            score = float(self._metric(x).item())
            # pyiqa DOVER returns a fused score; clamp to a sane range
            return max(0.0, min(1.0, score))
        except Exception:
            return None
