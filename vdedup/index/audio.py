"""Stage 4 (audio) — inverted hash index.

hash -> [(content_id, track, t)]. Probing is an exact dictionary lookup (cheap,
high-precision), and every shared hash yields a matched timestamp pair. IDF
weighting: a hash retrieving many distinct files (a common stinger, a near-
silence hash) is down-weighted by log(N_files / df).
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..descriptors.audio_fp import AudioFingerprint


class AudioIndex:
    def __init__(self):
        self.postings: dict[int, list[tuple[str, int, float]]] = defaultdict(list)
        self.files: set[str] = set()

    def add(self, content_id: str, fp: AudioFingerprint, track: int = 0) -> None:
        self.files.add(content_id)
        for h, t in zip(fp.hashes.tolist(), fp.times.tolist()):
            self.postings[h].append((content_id, track, t))

    def remove(self, content_id: str) -> None:
        self.files.discard(content_id)
        for h in list(self.postings):
            kept = [p for p in self.postings[h] if p[0] != content_id]
            if kept:
                self.postings[h] = kept
            else:
                del self.postings[h]

    def _df(self, h: int) -> int:
        return len({p[0] for p in self.postings.get(h, ())})

    def query(self, content_id: str, fp: AudioFingerprint, smoothing: float = 1.0):
        """Return {other_cid: list[(t_self, t_other, weight)]} from shared hashes."""
        n = max(len(self.files), 1)
        out: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for h, t_self in zip(fp.hashes.tolist(), fp.times.tolist()):
            posting = self.postings.get(h)
            if not posting:
                continue
            df = len({p[0] for p in posting if p[0] != content_id})
            if df == 0:
                continue
            w = math.log((n + smoothing) / (df + smoothing)) + 1e-3
            for other_cid, _track, t_other in posting:
                if other_cid != content_id:
                    out[other_cid].append((t_self, t_other, w))
        return out

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"postings": dict(self.postings), "files": self.files}, f)

    @classmethod
    def load(cls, path: str | Path) -> "AudioIndex":
        idx = cls()
        if Path(path).exists():
            with open(path, "rb") as f:
                d = pickle.load(f)
            idx.postings = defaultdict(list, d["postings"])
            idx.files = d["files"]
        return idx
