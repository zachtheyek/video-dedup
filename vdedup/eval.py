"""Benchmark harness: score a pipeline run against a ground-truth pair list, and
calibrate the TERRIBLE-gate thresholds to a library's own score distributions.

Ground-truth format (matches the validation file): duplicate paths in pairs,
separated by blank lines. Paths may be shell-escaped; matched by basename.

    path A
    path B
    <blank>
    path C
    path D
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s.strip())


def parse_ground_truth(path: str | Path) -> list[tuple[str, str]]:
    lines = Path(path).read_text().splitlines()
    pairs, buf = [], []
    for ln in lines:
        if ln.strip() == "":
            if len(buf) >= 2:
                pairs.append((buf[0], buf[1]))
            buf = []
        else:
            buf.append(Path(_unescape(ln)).name)
    if len(buf) >= 2:
        pairs.append((buf[0], buf[1]))
    return pairs


@dataclass
class EvalResult:
    recall: float
    n_pairs: int
    n_flagged: int
    pair_rows: list[dict] = field(default_factory=list)
    cross_group_merges: list[tuple[str, str]] = field(default_factory=list)
    n_clusters_multi: int = 0
    timings: dict = field(default_factory=dict)
    runtime: float = 0.0


def evaluate(result, catalog, gt_pairs: list[tuple[str, str]]) -> EvalResult:
    # filename -> content_id (including duplicate paths)
    fname_to_cid = {Path(r["path"]).name: r["content_id"] for r in catalog.iter_files()}
    for r in catalog.conn.execute("SELECT content_id, path FROM dup_path"):
        fname_to_cid[Path(r["path"]).name] = r["content_id"]
    cluster_of = {cid: c.cluster_id for c in result.clusters for cid in c.members}

    # ground-truth group id per file (each pair is its own group)
    group_of = {}
    for gi, (a, b) in enumerate(gt_pairs):
        group_of[a] = gi
        group_of[b] = gi

    rows, flagged = [], 0
    for a, b in gt_pairs:
        ca, cb = fname_to_cid.get(a), fname_to_cid.get(b)
        if ca is None or cb is None:
            rows.append(dict(a=a, b=b, status="MISSING-FILE", how=""))
            continue
        same_content = ca == cb
        same_cluster = ca in cluster_of and cluster_of.get(ca) == cluster_of.get(cb)
        ok = same_content or same_cluster
        flagged += int(ok)
        how = "exact-content-dup" if same_content else ("same-cluster" if same_cluster else "")
        rows.append(dict(a=a, b=b, status="PASS" if ok else "FAIL", how=how, ca=ca, cb=cb))

    # precision signal: two GT files from DIFFERENT pairs landing in one cluster
    cross = []
    for c in result.clusters:
        gt_groups = {}
        for cid in c.members:
            names = [n for n, x in fname_to_cid.items() if x == cid and n in group_of]
            for n in names:
                gt_groups.setdefault(group_of[n], []).append(n)
        if len(gt_groups) > 1:
            reps = [v[0] for v in gt_groups.values()]
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    cross.append((reps[i], reps[j]))

    n = len([r for r in rows if r["status"] != "MISSING-FILE"])
    return EvalResult(
        recall=flagged / n if n else 0.0, n_pairs=n, n_flagged=flagged, pair_rows=rows,
        cross_group_merges=cross,
        n_clusters_multi=sum(1 for c in result.clusters if len(c.members) > 1),
        timings=result.timings, runtime=sum(result.timings.values()))


def render_eval(ev: EvalResult) -> str:
    out = ["=" * 70, "BENCHMARK", "=" * 70,
           f"pair recall: {ev.n_flagged}/{ev.n_pairs} = {ev.recall:.0%}",
           f"multi-member clusters: {ev.n_clusters_multi}",
           f"cross-pair (precision) errors: {len(ev.cross_group_merges)}",
           f"runtime: {ev.runtime:.1f}s  " + "  ".join(f"{k}={v:.1f}s" for k, v in ev.timings.items()),
           "-" * 70]
    for r in ev.pair_rows:
        out.append(f"[{r['status']:>12}] {r.get('how','')}")
        out.append(f"     A: {r['a']}")
        out.append(f"     B: {r['b']}")
    if ev.cross_group_merges:
        out.append("-" * 70)
        out.append("CROSS-PAIR MERGES (different GT pairs clustered together):")
        for a, b in ev.cross_group_merges:
            out.append(f"   {a}  <>  {b}")
    return "\n".join(out)


# ---- calibration ----------------------------------------------------------
def calibrate_thresholds(quality_dicts: list[dict], low_pct: float = 5.0) -> dict:
    """Suggest TERRIBLE-gate floors from a library's own score distribution: a
    conservative low percentile of each metric, so only genuine outliers trip."""
    def col(key):
        vals = [q[key] for q in quality_dicts if q.get(key) is not None and q.get(key) > 0]
        return np.array(vals) if vals else np.array([0.0])
    reff, bw, dover, bpp = col("R_eff"), col("audio_bw_hz"), col("dover_tech"), col("bpp_norm")
    return {
        "terrible_reff_px": float(np.percentile(reff, low_pct)),
        "terrible_audio_bw_hz": float(np.percentile(bw, low_pct)),
        "terrible_dover": float(np.percentile(dover, low_pct)),
        "terrible_bpp_norm": float(np.percentile(bpp, low_pct)),
        "_stats": {k: dict(min=float(v.min()), p5=float(np.percentile(v, 5)),
                           p50=float(np.percentile(v, 50)), max=float(v.max()))
                   for k, v in [("R_eff", reff), ("audio_bw_hz", bw),
                                ("dover_tech", dover), ("bpp_norm", bpp)]},
    }
