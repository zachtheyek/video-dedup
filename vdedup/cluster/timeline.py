"""Stage 7 — global 1-D timeline reconstruction.

Lift pairwise offsets into one canonical axis per cluster. With alpha ~= 1 fixed,
each accepted edge (i, j) asserts  b_i - b_j = beta_ij. Stacking all edges gives
an over-determined system M.b = beta with M the signed incidence matrix; the
weighted least-squares solution (weights = inverse offset variance, so dense
sub-second audio edges dominate coarse vision-only ones) with the gauge fix
b_ref = 0 is exactly the weighted graph-Laplacian solve. The residual M.b - beta
localises inconsistent edges (cycle-consistency / over-merge detection).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Edge:
    i: str          # node id
    j: str          # node id
    beta: float     # offset such that  t_j = t_i + beta  (=> b_i - b_j = beta)
    weight: float = 1.0   # inverse-variance weight of this offset estimate


@dataclass
class TimelineSolution:
    offsets: dict[str, float]              # node id -> canonical offset b_i
    residuals: dict[tuple[str, str], float]  # edge (i,j) -> M.b - beta
    max_abs_residual: float
    intervals: dict[str, tuple[float, float]]  # node id -> (s_canonical, e_canonical)
    canonical_span: tuple[float, float]
    reference: str


def solve_timeline(nodes: list[str], edges: list[Edge],
                   extents: dict[str, tuple[float, float]],
                   reference: str | None = None) -> TimelineSolution:
    """Solve one connected component.

    `extents[node] = (t_local_min, t_local_max)` are the matched/used local time
    bounds of each file (seconds), used to place its canonical interval.
    `reference` pins the gauge (b_ref = 0); default = node with the widest extent
    (the longest / most complete member).
    """
    if not nodes:
        return TimelineSolution({}, {}, 0.0, {}, (0.0, 0.0), "")

    if reference is None:
        reference = max(nodes, key=lambda n: extents.get(n, (0.0, 0.0))[1] - extents.get(n, (0.0, 0.0))[0])

    idx = {n: k for k, n in enumerate(nodes)}
    N = len(nodes)

    if edges:
        E = len(edges)
        M = np.zeros((E, N))
        beta = np.zeros(E)
        w = np.zeros(E)
        for k, e in enumerate(edges):
            M[k, idx[e.i]] = 1.0
            M[k, idx[e.j]] = -1.0
            beta[k] = e.beta
            w[k] = max(e.weight, 1e-9)

        # gauge fix: drop the reference column (b_ref := 0)
        keep_cols = [k for k in range(N) if k != idx[reference]]
        Mr = M[:, keep_cols]
        sw = np.sqrt(w)
        # weighted least squares via row scaling
        A = Mr * sw[:, None]
        rhs = beta * sw
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        b = np.zeros(N)
        for col, val in zip(keep_cols, sol):
            b[col] = val
        # residuals in original (unweighted) units
        resid_vec = M @ b - beta
    else:
        b = np.zeros(N)
        resid_vec = np.zeros(0)

    offsets = {n: float(b[idx[n]]) for n in nodes}
    residuals = {(e.i, e.j): float(resid_vec[k]) for k, e in enumerate(edges)}
    max_resid = float(np.max(np.abs(resid_vec))) if len(resid_vec) else 0.0

    intervals: dict[str, tuple[float, float]] = {}
    for n in nodes:
        tmin, tmax = extents.get(n, (0.0, 0.0))
        intervals[n] = (tmin + offsets[n], tmax + offsets[n])

    S = min(v[0] for v in intervals.values())
    E_ = max(v[1] for v in intervals.values())
    return TimelineSolution(offsets, residuals, max_resid, intervals, (S, E_), reference)
