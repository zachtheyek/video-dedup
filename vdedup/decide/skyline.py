"""Stage 9 — the decision engine as a dominance skyline.

All six heuristics collapse into one partial order: keep every segment that is
not dominated by another. `a` dominates `b` iff `a` covers everything `b` covers
AND `a` is at least as preferred under the lexicographic priority
`(not terrible, is_full_length, quality)`. The kept set is the skyline; it is
order-independent, idempotent, and provably never opens a coverage gap (see the
coverage-preservation argument in the design doc, exercised by the fuzz test).

Policy knobs (all defaulting to the design's stated defaults):
  * rule6_quality_margin (Δ): keep a clip a full-length file would otherwise
    absorb, when the clip's quality exceeds the full file's by more than Δ.
  * audio_variant_keep_both: a containing segment does NOT absorb one carrying a
    distinct (non-empty, different) language — keeps one copy per language.
  * set_cover_pass: optional redundancy pass that drops a spanning segment fully
    covered by a *set* of individually-better segments (off by default —
    trades continuity for storage).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .segment import Segment


def covered_fraction(seg: Segment, lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 0:
        return 0.0
    return max(0.0, min(seg.e, hi) - max(seg.s, lo)) / span


def is_full(seg: Segment, lo: float, hi: float, eps: float = 0.05) -> bool:
    return covered_fraction(seg, lo, hi) >= 1.0 - eps


def priority(seg: Segment, lo: float, hi: float, eps: float) -> tuple[bool, bool, float]:
    return (not seg.terrible, is_full(seg, lo, hi, eps), seg.quality)


def better(a: Segment, b: Segment, lo: float, hi: float, eps: float = 0.05) -> int:
    """Total order: +1 if a is preferred, -1 if b is, 0 only if truly identical."""
    pa, pb = priority(a, lo, hi, eps), priority(b, lo, hi, eps)
    if pa != pb:
        return 1 if pa > pb else -1
    # deterministic tiebreak chain: quality, then audio, then codec, then
    # smaller file (favour storage), then stable id.
    for ka, kb in ((a.quality, b.quality),
                   (a.audio_quality, b.audio_quality),
                   (a.codec_rank, b.codec_rank),
                   (-a.size_bytes, -b.size_bytes),
                   (a.vid, b.vid)):
        if ka != kb:
            return 1 if ka > kb else -1
    return 0


def contains(a: Segment, b: Segment, tol: float = 0.5) -> bool:
    return a.s <= b.s + tol and a.e >= b.e - tol


def dominates(a: Segment, b: Segment, lo: float, hi: float, *,
              eps: float = 0.05, tol: float = 0.5,
              rule6_quality_margin: float | None = None,
              audio_variant_keep_both: bool = True) -> bool:
    if a.vid == b.vid:
        return False
    if not (contains(a, b, tol) and better(a, b, lo, hi, eps) >= 0):
        return False
    # Audio-variant policy: a distinct-language copy is never absorbed.
    if audio_variant_keep_both and a.lang and b.lang and a.lang != b.lang:
        return False
    # Rule-6 Δ override: if a wins purely on the full-length bit but b is much
    # higher quality, keep b.
    if rule6_quality_margin is not None:
        a_full = is_full(a, lo, hi, eps)
        b_full = is_full(b, lo, hi, eps)
        same_terrible = (not a.terrible) == (not b.terrible)
        if same_terrible and a_full and not b_full and (b.quality - a.quality) > rule6_quality_margin:
            return False
    return True


@dataclass
class PruneResult:
    keep: list[Segment]
    drop: list[Segment]
    dominated_by: dict[str, str] = field(default_factory=dict)  # dropped vid -> dominating vid

    @property
    def keep_ids(self) -> set[str]:
        return {s.vid for s in self.keep}


def prune(cluster: list[Segment], *, eps: float = 0.05, contain_tol: float = 0.5,
          rule6_quality_margin: float | None = None,
          audio_variant_keep_both: bool = True,
          set_cover_pass: bool = False,
          span: tuple[float, float] | None = None) -> PruneResult:
    """Compute the kept skyline of a cluster.

    `span` overrides the canonical [S, E] (e.g. from the timeline solve or an
    external runtime); otherwise it is the hull of the members' intervals.
    """
    if not cluster:
        return PruneResult([], [])
    lo, hi = span if span is not None else (min(s.s for s in cluster), max(s.e for s in cluster))

    dominated_by: dict[str, str] = {}
    keep: list[Segment] = []
    for x in cluster:
        dominator = None
        for y in cluster:
            if y.vid == x.vid:
                continue
            if dominates(y, x, lo, hi, eps=eps, tol=contain_tol,
                         rule6_quality_margin=rule6_quality_margin,
                         audio_variant_keep_both=audio_variant_keep_both):
                dominator = y.vid
                break
        if dominator is None:
            keep.append(x)
        else:
            dominated_by[x.vid] = dominator

    if set_cover_pass:
        keep, extra = _set_cover_redundancy(keep, lo, hi, eps)
        for seg in extra:
            # mark covered-by-set; dominator recorded as the union sentinel
            dominated_by[seg.vid] = "<set-cover>"

    keep_ids = {x.vid for x in keep}
    drop = [x for x in cluster if x.vid not in keep_ids]
    return PruneResult(keep, drop, dominated_by)


def _set_cover_redundancy(keep: list[Segment], lo: float, hi: float, eps: float
                          ) -> tuple[list[Segment], list[Segment]]:
    """Optional pass: drop a kept segment whose whole interval is covered by the
    *union* of other kept segments that are each individually >= it in priority.

    Conservative: only drops a segment if the rest of the keep set still covers
    every point it covered (so coverage is preserved) and each covering segment
    is better_or_equal. Processed worst-first so the spanning/continuity files
    survive ties.
    """
    order = sorted(keep, key=lambda s: priority(s, lo, hi, eps))  # worst first
    survivors = list(keep)
    removed: list[Segment] = []
    for seg in order:
        others = [o for o in survivors if o.vid != seg.vid
                  and better(o, seg, lo, hi, eps) >= 0]
        if _covers_interval(others, seg.s, seg.e):
            survivors = [o for o in survivors if o.vid != seg.vid]
            removed.append(seg)
    return survivors, removed


def _covers_interval(segs: list[Segment], s: float, e: float, tol: float = 0.5) -> bool:
    """Do `segs` jointly cover [s, e] with no gap larger than tol?"""
    ivals = sorted((max(seg.s, s), min(seg.e, e)) for seg in segs
                   if seg.e > s and seg.s < e)
    cursor = s
    for a, b in ivals:
        if a > cursor + tol:
            return False
        cursor = max(cursor, b)
        if cursor >= e - tol:
            return True
    return cursor >= e - tol
