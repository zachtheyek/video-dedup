"""Stage 10 — dry-run audit report.

Every proposed prune is annotated with its dominator and the supporting evidence
(canonical intervals, quality scores, alignment modality). Nothing is deleted by
rendering a report.
"""
from __future__ import annotations


def _fmt_seg(seg) -> str:
    flags = []
    if seg.terrible:
        flags.append("TERRIBLE")
    if seg.lang:
        flags.append(f"lang={seg.lang}")
    tag = (" [" + ",".join(flags) + "]") if flags else ""
    return (f"[{seg.s:7.2f},{seg.e:7.2f}]  Q={seg.quality:6.3f}  "
            f"Qa={seg.audio_quality:5.2f}  {_short(seg.vid)}{tag}")


def _short(cid: str) -> str:
    return cid[:10]


def render_report(result, catalog) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"vdedup run {result.run_id}   files={result.n_files}  "
                 f"exact-dups={result.n_duplicates}  clusters={len(result.clusters)}  "
                 f"proposed-prunes={len(result.prune_ids)}")
    lines.append("=" * 78)

    multi = [c for c in result.clusters if len(c.members) > 1]
    singles = [c for c in result.clusters if len(c.members) == 1]

    for c in sorted(multi, key=lambda c: -len(c.members)):
        flag = "  *** NEEDS REVIEW ***" if c.needs_review else ""
        lines.append("")
        lines.append(f"Cluster {c.cluster_id}  span=[{c.canonical_span[0]:.1f},"
                     f"{c.canonical_span[1]:.1f}]s  members={len(c.members)}  "
                     f"resid={c.solve_residual:.2f}s{flag}")
        for cid in c.keep:
            lines.append(f"   KEEP   {_fmt_seg(c.segments[cid])}")
        for cid in c.drop:
            dom = c.dominated_by.get(cid)
            dom_s = _short(dom) if dom else "?"
            lines.append(f"   prune  {_fmt_seg(c.segments[cid])}  <- dominated by {dom_s}")

    if singles:
        lines.append("")
        lines.append(f"{len(singles)} singleton title(s) (no redundancy, all kept).")

    if result.review_pairs:
        lines.append("")
        lines.append(render_review(result))

    lines.append("")
    lines.append(f"DRY RUN — nothing deleted. {len(result.prune_ids)} file(s) would be quarantined.")
    return "\n".join(lines)


def render_review(result) -> str:
    if not result.review_pairs:
        return "Review queue: empty."
    out = [f"Review queue ({len(result.review_pairs)} pair(s) — not auto-merged):"]
    for a, b, reason in result.review_pairs:
        out.append(f"   {a[:10]} ~ {b[:10]}   {reason}")
    return "\n".join(out)
