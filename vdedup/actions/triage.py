"""Minimal interactive triage for the review queue of the most recent run.

Reads review-flagged edges from the catalog (audio-only matches, piecewise/
different-cut and A/V-desync flags) and walks the user through them, recording a
verdict per pair to <data_dir>/review_decisions.json. Read-only w.r.t. the media
— it records human decisions, it does not move files."""
from __future__ import annotations

import json
from pathlib import Path

import click

from ..catalog import Catalog


def _review_edges(catalog: Catalog):
    return catalog.conn.execute(
        "SELECT a,b,modality,beta,a_inliers,v_inliers,span_seconds,confidence "
        "FROM match_edge WHERE route_review=1 OR modality='audio' ORDER BY confidence DESC"
    ).fetchall()


def run_triage(cfg) -> None:
    db = Path(cfg.data_dir) / "catalog.sqlite"
    if not db.exists():
        click.echo("No catalog yet — run `vdedup scan` first.")
        return
    cat = Catalog(db)
    name = {r["content_id"]: Path(r["path"]).name for r in cat.iter_files()}
    edges = _review_edges(cat)
    if not edges:
        click.echo("Review queue is empty.")
        cat.close()
        return

    out_path = Path(cfg.data_dir) / "review_decisions.json"
    decisions = json.loads(out_path.read_text()) if out_path.exists() else {}
    click.echo(f"{len(edges)} pair(s) to review.\n")
    for e in edges:
        key = f"{e['a']}|{e['b']}"
        if key in decisions:
            continue
        click.echo("-" * 60)
        click.echo(f"  A: {name.get(e['a'], e['a'])}")
        click.echo(f"  B: {name.get(e['b'], e['b'])}")
        click.echo(f"  modality={e['modality']} offset={e['beta']:.1f}s "
                   f"audio_inliers={e['a_inliers']} video_inliers={e['v_inliers']} "
                   f"span={e['span_seconds']:.0f}s conf={e['confidence']:.2f}")
        v = click.prompt("  [d]uplicate / [k]eep both / [s]kip / [q]uit", default="s")
        if v == "q":
            break
        decisions[key] = {"d": "duplicate", "k": "keep-both", "s": "skip"}.get(v, "skip")
    out_path.write_text(json.dumps(decisions, indent=2))
    click.echo(f"\nSaved verdicts to {out_path}")
    cat.close()
