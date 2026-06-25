"""Stage 10 — reversible actions: quarantine + manifest + TTL.

No destructive action without a manifest. Pruned files (and the redundant paths
of exact-content duplicates) move to a quarantine directory with a JSON manifest
mapping each quarantined file back to its original path, cluster, dominator and
run id, so every action is reversible within the TTL window and fully auditable.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class QuarantineItem:
    path: str
    content_id: str
    cluster_id: str
    reason: str
    dominated_by: str | None
    original_path: str          # = path (kept for symmetry with restore)


@dataclass
class PrunePlan:
    run_id: str
    items: list[QuarantineItem] = field(default_factory=list)
    keeps: list[tuple[str, str]] = field(default_factory=list)   # (content_id, path)
    review: list[tuple[str, str, str]] = field(default_factory=list)


def build_plan(result, catalog) -> PrunePlan:
    plan = PrunePlan(run_id=result.run_id, review=result.review_pairs)
    for c in result.clusters:
        for cid in c.drop:
            for p in catalog.get_paths_for_content(cid):
                plan.items.append(QuarantineItem(
                    path=p, content_id=cid, cluster_id=c.cluster_id,
                    reason="dominated", dominated_by=c.dominated_by.get(cid), original_path=p))
        for cid in c.keep:
            rep = (catalog.get_file(cid) or {})["path"]
            plan.keeps.append((cid, rep))
            for p in catalog.get_paths_for_content(cid):
                if p != rep:
                    plan.items.append(QuarantineItem(
                        path=p, content_id=cid, cluster_id=c.cluster_id,
                        reason="exact-content-duplicate", dominated_by=rep, original_path=p))
    return plan


def apply_plan(plan: PrunePlan, quarantine_dir: str | Path, *, move: bool = True) -> Path:
    """Move quarantined files under <quarantine_dir>/<run_id>/ and write manifest.
    Returns the manifest path. Skips files whose dominator/keep is missing on disk
    (never strand a file with no surviving copy)."""
    qroot = Path(quarantine_dir) / plan.run_id
    qroot.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": plan.run_id, "created_at": time.time(), "items": []}
    for i, item in enumerate(plan.items):
        src = Path(item.path)
        rec = asdict(item)
        if not src.exists():
            rec["status"] = "missing-source"
            manifest["items"].append(rec)
            continue
        dest = qroot / f"{i:04d}_{src.name}"
        if move:
            shutil.move(str(src), str(dest))
        else:
            shutil.copy2(str(src), str(dest))
        rec["quarantine_path"] = str(dest)
        rec["status"] = "moved" if move else "copied"
        manifest["items"].append(rec)
    mpath = qroot / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    return mpath


def restore(manifest_path: str | Path) -> int:
    """Move quarantined files back to their original paths. Returns count restored."""
    m = json.loads(Path(manifest_path).read_text())
    n = 0
    for item in m["items"]:
        q = item.get("quarantine_path")
        if q and Path(q).exists():
            Path(item["original_path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(q, item["original_path"])
            n += 1
    return n


def purge_expired(quarantine_dir: str | Path, ttl_days: int) -> list[str]:
    """Delete quarantine run directories older than ttl_days. Returns purged dirs."""
    root = Path(quarantine_dir)
    if not root.exists():
        return []
    cutoff = time.time() - ttl_days * 86400
    purged = []
    for run_dir in root.iterdir():
        mpath = run_dir / "manifest.json"
        if mpath.exists():
            created = json.loads(mpath.read_text()).get("created_at", 0)
            if created < cutoff:
                shutil.rmtree(run_dir)
                purged.append(str(run_dir))
    return purged
