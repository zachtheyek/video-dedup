"""Stage 10 — dry-run report.

Two renderings of the same result:
  * default      — plain-language, filenames, summarised; readable by anyone.
  * verbose=True — the technical view (content-ids, quality scores, canonical
                   intervals, alignment modality, the full review queue).
Nothing is deleted by rendering a report.
"""
from __future__ import annotations

from pathlib import Path


def _names(catalog) -> dict:
    return {r["content_id"]: Path(r["path"]).name for r in catalog.iter_files()}


def _mins(span) -> str:
    m = max(0.0, span[1] - span[0]) / 60.0
    return f"{m:.0f} min" if m >= 1 else f"{int(m * 60)} s"


# --------------------------------------------------------------------------
# default (human-friendly)
# --------------------------------------------------------------------------
def render_report(result, catalog, verbose: bool = False) -> str:
    if verbose:
        return _render_verbose(result, catalog)

    name = _names(catalog)
    multi = sorted((c for c in result.clusters if len(c.members) > 1),
                   key=lambda c: -len(c.members))
    singles = [c for c in result.clusters if len(c.members) == 1]
    n_remove = len(result.prune_ids)

    out = []
    out.append(f"Scanned {result.n_files} file(s).")
    if result.n_duplicates:
        out.append(f"{result.n_duplicates} exact duplicate file(s) found at ingest.")
    out.append(f"Found {len(multi)} group(s) of duplicate / related files; "
               f"{n_remove} file(s) can be removed.")
    out.append("")

    for i, c in enumerate(multi, 1):
        rev = "  (flagged for review — alignment looked inconsistent)" if c.needs_review else ""
        out.append(f"── Group {i} — {len(c.members)} files, ~{_mins(c.canonical_span)}{rev}")
        for cid in c.keep:
            seg = c.segments[cid]
            note = " — low quality" if seg.terrible else ""
            extra = " (kept — covers part the others don't)" if len(c.keep) > 1 else ""
            out.append(f"     keep    {name.get(cid, cid)}{note}{extra}")
        for cid in c.drop:
            dom = c.dominated_by.get(cid)
            why = f'duplicate of "{name.get(dom, dom)}"' if dom else "duplicate"
            out.append(f"     REMOVE  {name.get(cid, cid)}   ({why})")
        out.append("")

    if singles:
        out.append(f"{len(singles)} file(s) have no duplicates — all kept.")
    if result.review_pairs:
        out.append("")
        out.append(f"{len(result.review_pairs)} similar pair(s) need manual review "
                   f"(look-alike but not confirmed duplicates).")
        out.append("   → run `vdedup review` to go through them"
                   " (or re-run with --verbose to see them all).")
    out.append("")
    if n_remove:
        out.append(f"DRY RUN — nothing deleted. Run `vdedup apply <dir>` to move the "
                   f"{n_remove} removable file(s) to quarantine (reversible).")
    else:
        out.append("DRY RUN — no removable duplicates found.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# verbose (technical)
# --------------------------------------------------------------------------
def _render_verbose(result, catalog) -> str:
    name = _names(catalog)

    def seg_line(seg):
        flags = (["TERRIBLE"] if seg.terrible else []) + ([f"lang={seg.lang}"] if seg.lang else [])
        tag = (" [" + ",".join(flags) + "]") if flags else ""
        return (f"[{seg.s:8.2f},{seg.e:8.2f}] Q={seg.quality:6.3f} Qa={seg.audio_quality:5.2f} "
                f"{seg.vid[:10]} {name.get(seg.vid, '')}{tag}")

    lines = ["=" * 78,
             f"vdedup run {result.run_id}  files={result.n_files} exact-dups={result.n_duplicates} "
             f"clusters={len(result.clusters)} active={result.n_active} prunes={len(result.prune_ids)}",
             "timings: " + "  ".join(f"{k}={v:.1f}s" for k, v in result.timings.items()),
             "=" * 78]
    for c in sorted((c for c in result.clusters if len(c.members) > 1), key=lambda c: -len(c.members)):
        flag = "  *** NEEDS REVIEW ***" if c.needs_review else ""
        lines.append("")
        lines.append(f"Cluster {c.cluster_id}  span=[{c.canonical_span[0]:.1f},"
                     f"{c.canonical_span[1]:.1f}]s  members={len(c.members)}  "
                     f"resid={c.solve_residual:.2f}s{flag}")
        for cid in c.keep:
            lines.append(f"   KEEP   {seg_line(c.segments[cid])}")
        for cid in c.drop:
            dom = c.dominated_by.get(cid)
            lines.append(f"   prune  {seg_line(c.segments[cid])}  <- {name.get(dom, dom)}")
    singles = [c for c in result.clusters if len(c.members) == 1]
    if singles:
        lines.append("")
        lines.append(f"{len(singles)} singleton(s) (no redundancy, all kept).")
    if result.review_pairs:
        lines.append("")
        lines.append(f"Review queue ({len(result.review_pairs)} pair(s) — not auto-merged):")
        for a, b, reason in result.review_pairs:
            lines.append(f"   {name.get(a, a[:10])} ~ {name.get(b, b[:10])}   {reason}")
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
