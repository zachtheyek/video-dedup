"""Content-addressed feature cache.

Per-file features (frame embeddings, pHashes, frame times, audio landmark
hashes) are computed once per file *ever* and cached on disk keyed by
content_id, so incremental re-scans never recompute them. Stored as compact
.npz blobs under <data>/cache/<cid[:2]>/<cid>/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatureCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _dir(self, content_id: str) -> Path:
        return self.root / content_id[:2] / content_id

    def has(self, content_id: str, name: str) -> bool:
        return (self._dir(content_id) / f"{name}.npz").exists()

    def save(self, content_id: str, name: str, **arrays: np.ndarray) -> None:
        d = self._dir(content_id)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{name}.npz", **arrays)

    def load(self, content_id: str, name: str) -> dict[str, np.ndarray] | None:
        p = self._dir(content_id) / f"{name}.npz"
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def delete(self, content_id: str) -> None:
        import shutil
        d = self._dir(content_id)
        if d.exists():
            shutil.rmtree(d)
