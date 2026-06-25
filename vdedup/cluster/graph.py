"""Stage 6 — clustering: connected components + cycle-consistency validation.

Only video-grounded edges (the `both`, `visual`, and `audio_variant` rows of the
Section-8 decision table) form clusters; audio-only matches are held out for
review. Connected components are the candidate clusters; the timeline solve's
per-edge residuals then localise inconsistent edges, which are dropped and the
component re-evaluated for connectivity (over-merge / hub mitigation).
"""
from __future__ import annotations

from .timeline import Edge, TimelineSolution


def connected_components(nodes: list[str], edges: list[Edge]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        if e.i in parent and e.j in parent:
            union(e.i, e.j)

    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    # stable, largest-first ordering
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def cycle_check(solution: TimelineSolution, tol: float) -> list[tuple[str, str]]:
    """Return edges whose offset residual exceeds `tol` seconds — probable false
    matches to drop before re-checking connectivity."""
    return [edge for edge, r in solution.residuals.items() if abs(r) > tol]
