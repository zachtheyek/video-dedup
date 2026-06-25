"""Stage 4 (visual) — ANN index over frame descriptors.

Two backends behind one interface:
  * "embedding": SSCD vectors in a FAISS inner-product index (cosine, since
    vectors are L2-normalised); falls back to a numpy brute-force IP if faiss
    is unavailable.
  * "phash": 64-bit hashes matched by Hamming distance (brute-force popcount).
Used when SSCD weights are unavailable.

Either backend returns, per query file, matched {(t_self, t_other)} pairs with an
IDF weight (a query frame whose near-neighbours span many distinct files is
uninformative boilerplate and is down-weighted).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..descriptors.phash import hamming


class VisualIndex:
    def __init__(self, mode: str = "embedding", dim: int = 512,
                 sim_min: float = 0.6, hamming_max: int = 10, knn_k: int = 10,
                 use_faiss: bool = False):
        # use_faiss defaults False: on macOS, faiss-cpu and torch both link
        # OpenMP and deadlock when used in the same process (a 0%-CPU hang). The
        # exact numpy inner-product path is robust and fast at single-host scale;
        # enable faiss only on Linux / very large corpora.
        self.mode = mode
        self.dim = dim
        self.sim_min = sim_min
        self.hamming_max = hamming_max
        self.knn_k = knn_k
        self.use_faiss = use_faiss
        self._vecs: list[np.ndarray] = []
        self._hashes: list[np.ndarray] = []
        self.cids: list[str] = []
        self.times: list[float] = []
        self.files: set[str] = set()
        self._index = None
        self._mat = None
        self._hmat = None

    def add(self, content_id: str, primitives: np.ndarray, times: np.ndarray) -> None:
        if primitives.shape[0] == 0:
            self.files.add(content_id)
            return
        self.files.add(content_id)
        if self.mode == "embedding":
            self._vecs.append(primitives.astype(np.float32))
        else:
            self._hashes.append(primitives.astype(np.uint64))
        self.cids.extend([content_id] * primitives.shape[0])
        self.times.extend(times.tolist())

    def build(self) -> None:
        self.cids_arr = np.array(self.cids)
        self.times_arr = np.array(self.times)
        if self.mode == "embedding":
            self._mat = np.concatenate(self._vecs, axis=0) if self._vecs else np.zeros((0, self.dim), np.float32)
            if self.use_faiss:
                try:
                    import faiss
                    faiss.omp_set_num_threads(1)
                    idx = faiss.IndexFlatIP(self.dim)
                    if self._mat.shape[0]:
                        idx.add(self._mat)
                    self._index = idx
                except Exception:
                    self._index = None
        else:
            self._hmat = np.concatenate(self._hashes, axis=0) if self._hashes else np.zeros((0,), np.uint64)

    def _neighbors_embedding(self, vecs: np.ndarray):
        k = min(self.knn_k, len(self.cids_arr)) or 1
        if self._index is not None:
            D, I = self._index.search(vecs.astype(np.float32), k)
        else:  # numpy brute-force inner product
            sims = vecs.astype(np.float32) @ self._mat.T
            I = np.argsort(-sims, axis=1)[:, :k]
            D = np.take_along_axis(sims, I, axis=1)
        return D, I

    def query(self, content_id: str, primitives: np.ndarray, times: np.ndarray):
        out: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        if primitives.shape[0] == 0:
            return out
        n_files = max(len(self.files), 1)

        if self.mode == "embedding":
            D, I = self._neighbors_embedding(primitives.astype(np.float32))
            for qi in range(primitives.shape[0]):
                hits = [(self.cids_arr[j], self.times_arr[j], float(D[qi, r]))
                        for r, j in enumerate(I[qi]) if D[qi, r] >= self.sim_min
                        and self.cids_arr[j] != content_id]
                if not hits:
                    continue
                df = len({c for c, _, _ in hits})
                w = math.log((n_files + 1) / (df + 1)) + 1e-3
                for c, t_other, _s in hits:
                    out[c].append((float(times[qi]), float(t_other), w))
        else:
            for qi in range(primitives.shape[0]):
                d = hamming(self._hmat, np.uint64(primitives[qi]))
                near = np.where(d <= self.hamming_max)[0]
                hits = [(self.cids_arr[j], self.times_arr[j]) for j in near
                        if self.cids_arr[j] != content_id]
                if not hits:
                    continue
                df = len({c for c, _ in hits})
                w = math.log((n_files + 1) / (df + 1)) + 1e-3
                for c, t_other in hits:
                    out[c].append((float(times[qi]), float(t_other), w))
        return out
