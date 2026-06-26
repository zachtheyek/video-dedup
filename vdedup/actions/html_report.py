"""Self-contained HTML report with a thumbnail per file, keeps in green, prunes
in red with their dominator. One frame per file is extracted at the file's
canonical midpoint."""
from __future__ import annotations

import html
import subprocess
from pathlib import Path


def _thumb(src: str, dest: Path, t: float) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.2f}", "-i", src,
                        "-frames:v", "1", "-vf", "scale=200:-1", "-y", str(dest)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0 and dest.exists()


def write_html(result, catalog, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    thumbs = out_path.parent / (out_path.stem + "_thumbs")
    name = {r["content_id"]: Path(r["path"]).name for r in catalog.iter_files()}
    path = {r["content_id"]: r["path"] for r in catalog.iter_files()}

    def card(cid, seg, kind, dom=None):
        mid = (seg.s + seg.e) / 2 if seg else 0
        tname = f"{cid[:12]}.jpg"
        ok = _thumb(path.get(cid, ""), thumbs / tname, mid) if cid in path else False
        img = (f'<img src="{thumbs.name}/{tname}">' if ok
               else '<div class="noimg">no preview</div>')
        meta = ""
        if seg:
            meta = (f"<small>[{seg.s:.0f}–{seg.e:.0f}s] Q={seg.quality:.2f}"
                    + (f" {seg.lang}" if seg.lang else "")
                    + (" TERRIBLE" if seg.terrible else "") + "</small>")
        domline = f'<small class="dom">↓ {html.escape(name.get(dom, "")[:40])}</small>' if dom else ""
        return (f'<div class="card {kind}">{img}<div class="t">{html.escape(name.get(cid,cid)[:48])}</div>'
                f'{meta}{domline}</div>')

    rows = []
    multi = [c for c in result.clusters if len(c.members) > 1]
    for c in sorted(multi, key=lambda c: -len(c.members)):
        flag = ' <span class="rev">NEEDS REVIEW</span>' if c.needs_review else ""
        cards = "".join(card(cid, c.segments.get(cid), "keep") for cid in c.keep)
        cards += "".join(card(cid, c.segments.get(cid), "prune", c.dominated_by.get(cid)) for cid in c.drop)
        rows.append(f'<section><h3>Cluster · span [{c.canonical_span[0]:.0f}–{c.canonical_span[1]:.0f}]s '
                    f'· {len(c.members)} files{flag}</h3><div class="grid">{cards}</div></section>')

    review = ""
    if result.review_pairs:
        items = "".join(f"<li>{html.escape(name.get(a,a)[:40])} ~ {html.escape(name.get(b,b)[:40])} "
                        f"<small>{html.escape(r)}</small></li>" for a, b, r in result.review_pairs)
        review = f"<section><h3>Review queue ({len(result.review_pairs)})</h3><ul>{items}</ul></section>"

    style = """body{font-family:-apple-system,sans-serif;margin:24px;background:#111;color:#eee}
    h1{font-size:20px}.grid{display:flex;flex-wrap:wrap;gap:10px}
    .card{width:200px;border-radius:8px;padding:6px;background:#1c1c1c}
    .card img{width:200px;border-radius:4px;display:block}
    .noimg{width:200px;height:112px;background:#333;display:flex;align-items:center;justify-content:center;color:#888}
    .keep{border:2px solid #2ecc71}.prune{border:2px solid #e74c3c;opacity:.85}
    .t{font-size:12px;margin-top:4px;word-break:break-word}small{color:#aaa;font-size:11px;display:block}
    .dom{color:#e67e22}.rev{color:#f1c40f}section{margin:18px 0}"""
    body = (f"<h1>vdedup report — {result.n_files} files, {len(result.prune_ids)} proposed prunes "
            f"(DRY RUN)</h1>" + "".join(rows) + review)
    out_path.write_text(f"<!doctype html><html><head><meta charset=utf-8><style>{style}</style></head>"
                        f"<body>{body}</body></html>")
    return out_path
