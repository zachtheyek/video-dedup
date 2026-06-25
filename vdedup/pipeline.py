"""End-to-end orchestration: scan -> features -> index -> match -> cluster ->
timeline -> decide. Produces a reversible, dry-run-by-default plan.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .catalog import Catalog, FeatureCache
from .features import FeatureExtractor
from .ingest import scan_tree
from .ingest.probe import parse_probe
from .descriptors.audio_fp import AudioFingerprint
from .descriptors.embed import Embedder
from .index import AudioIndex, VisualIndex, generate_candidates
from .align import verify_pair
from .cluster import connected_components, solve_timeline, Edge, cycle_check
from .decide import Segment
from .decide.skyline import prune

_CODEC_RANK = {"av1": 5, "hevc": 4, "h265": 4, "vp9": 3, "h264": 2, "mpeg4": 1}
_VIDEO_GROUNDED = {"both", "visual", "audio_variant"}


@dataclass
class ClusterDecision:
    cluster_id: str
    members: list[str]
    keep: list[str]
    drop: list[str]
    dominated_by: dict[str, str]
    canonical_span: tuple[float, float]
    solve_residual: float
    needs_review: bool
    segments: dict[str, Segment] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    clusters: list[ClusterDecision]
    review_pairs: list[tuple[str, str, str]]   # (a, b, reason)
    edges: dict[tuple[str, str], dict]
    n_files: int
    n_duplicates: int

    @property
    def prune_ids(self) -> list[str]:
        return [d for c in self.clusters for d in c.drop]


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.data_path.mkdir(parents=True, exist_ok=True)
        self.catalog = Catalog(cfg.data_path / "catalog.sqlite")
        self.cache = FeatureCache(cfg.data_path / "cache")
        self.embedder = Embedder(device=cfg.device, embed_size=cfg.vision.embed_size,
                                 models_dir=cfg.models_dir) if cfg.vision.use_sscd else None
        self.fx = FeatureExtractor(cfg, self.cache, self.embedder)

    # ---- public ----------------------------------------------------------
    def run(self, root: str | Path | None = None) -> RunResult:
        root = root or self.cfg.root
        n_dups = self._scan(root)
        cids = self.catalog.all_content_ids()
        feats = self._extract_all(cids)
        edges, edge_records, review = self._match(cids, feats)
        clusters = self._cluster_and_decide(cids, edges, edge_records)
        review = self._filter_review(review, clusters)
        run_id = uuid.uuid4().hex[:12]
        self._persist(run_id, clusters, edge_records)
        return RunResult(run_id, clusters, review, edge_records, len(cids), n_dups)

    @staticmethod
    def _filter_review(review, clusters):
        """Keep only review pairs whose members were NOT already merged into the
        same cluster — those are the genuinely ambiguous ones (e.g. audio shared
        across different artifacts). Intra-cluster pairs are already handled."""
        member_of = {}
        for c in clusters:
            for cid in c.members:
                member_of[cid] = c.cluster_id
        out = []
        seen = set()
        for a, b, reason in review:
            if member_of.get(a) is not None and member_of.get(a) == member_of.get(b):
                continue
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b, reason))
        return out

    # ---- stage 1 ---------------------------------------------------------
    def _scan(self, root) -> int:
        n_dups = 0
        for res in scan_tree(root, self.catalog, self.cfg):
            if res.status == "duplicate":
                n_dups += 1
        self.catalog.commit()
        return n_dups

    def _info_crop(self, cid: str):
        row = self.catalog.get_file(cid)
        info = parse_probe(json.loads(row["probe_json"]))
        sm = self.catalog.get_stream_meta(cid)
        crop = json.loads(sm["active_crop"]) if sm and sm["active_crop"] else None
        crop = tuple(crop) if crop else None
        return row, info, crop

    # ---- stages 2-3 + 8 --------------------------------------------------
    def _extract_all(self, cids: list[str]) -> dict[str, "object"]:
        feats = {}
        for cid in cids:
            row, info, crop = self._info_crop(cid)
            path = row["path"]
            feats[cid] = self.fx.extract(cid, path, info, crop)
            if self.catalog.get_quality(cid) is None:
                q = self.fx.score(cid, path, info, crop, feats[cid].has_audio)
                self.catalog.set_quality(cid, q.to_dict())
        self.catalog.commit()
        return feats

    # ---- stages 4-5 ------------------------------------------------------
    def _build_indexes(self, cids, feats):
        a = self.cfg
        mode = "embedding" if any(feats[c].mode == "embedding" for c in cids) else "phash"
        aidx = AudioIndex()
        vidx = VisualIndex(mode=mode, sim_min=a.candidate.visual_sim_min,
                           knn_k=a.candidate.knn_k, use_faiss=a.candidate.use_faiss)
        for c in cids:
            f = feats[c]
            aidx.add(c, AudioFingerprint(f.ahashes, f.atimes))
            prim = f.vecs if mode == "embedding" else f.phash
            vidx.add(c, prim, f.vtimes)
        vidx.build()
        return aidx, vidx, mode

    def _match(self, cids, feats):
        aidx, vidx, mode = self._build_indexes(cids, feats)
        al = self.cfg.align
        grid = (np.arange(*self._grid()).tolist() if al.scale_search else (1.0,))
        edges: list[Edge] = []
        records: dict[tuple[str, str], dict] = {}
        review: list[tuple[str, str, str]] = []
        processed: set[frozenset] = set()
        for a in cids:
            f = feats[a]
            am = aidx.query(a, AudioFingerprint(f.ahashes, f.atimes), self.cfg.candidate.idf_smoothing)
            prim = f.vecs if mode == "embedding" else f.phash
            vm = vidx.query(a, prim, f.vtimes)
            for cand in generate_candidates(a, am, vm, self.cfg):
                b = cand.other
                key = frozenset((a, b))
                if key in processed:
                    continue
                processed.add(key)
                dec = verify_pair(cand.pairs, has_audio_a=f.has_audio,
                                  has_audio_b=feats[b].has_audio, cfg=al, alpha_grid=grid)
                rec = dict(alpha=dec.alpha, beta=dec.beta, v_inliers=dec.v_inliers,
                           a_inliers=dec.a_inliers, span_seconds=dec.span_seconds,
                           residual_std=dec.residual_std, modality=dec.modality,
                           audio_agrees=dec.audio_agrees, confidence=dec.confidence,
                           route_review=dec.route_review, piecewise=dec.piecewise)
                records[(a, b)] = rec
                if dec.accept and dec.modality in _VIDEO_GROUNDED:
                    w = (dec.a_inliers / al.audio_offset_var + dec.v_inliers / al.visual_offset_var)
                    edges.append(Edge(a, b, dec.beta, max(w, 1e-6)))
                    if dec.route_review:
                        review.append((a, b, f"{dec.modality}: {dec.reason}"))
                elif dec.modality == "audio" or dec.route_review:
                    review.append((a, b, dec.reason))
        return edges, records, review

    def _grid(self):
        lo, hi, step = self.cfg.align.alpha_grid
        return (lo, hi + step / 2, step)

    # ---- stages 6-7 + 9 --------------------------------------------------
    def _cluster_and_decide(self, cids, edges, records) -> list[ClusterDecision]:
        out = []
        for comp in connected_components(cids, edges):
            comp_edges = [e for e in edges if e.i in comp and e.j in comp]
            for sub_members, sub_edges, resid in self._solve_with_cycle_check(comp, comp_edges):
                out.append(self._decide_cluster(sub_members, sub_edges, resid))
        return out

    def _solve_with_cycle_check(self, members, edges):
        extents = {c: self._extent(c) for c in members}
        sol = solve_timeline(members, edges, extents)
        bad = cycle_check(sol, self.cfg.align.cycle_residual_tol)
        if not bad:
            yield members, edges, sol.max_abs_residual
            return
        kept = [e for e in edges if (e.i, e.j) not in bad and (e.j, e.i) not in bad]
        for comp in connected_components(members, kept):
            ce = [e for e in kept if e.i in comp and e.j in comp]
            s = solve_timeline(comp, ce, {c: extents[c] for c in comp})
            yield comp, ce, s.max_abs_residual

    def _extent(self, cid):
        sm = self.catalog.get_stream_meta(cid)
        dur = (sm["declared_duration"] if sm and sm["declared_duration"] else 0.0) or 0.0
        return (0.0, float(dur))

    def _decide_cluster(self, members, edges, resid) -> ClusterDecision:
        extents = {c: self._extent(c) for c in members}
        sol = solve_timeline(members, edges, extents)
        d = self.cfg.decide
        segs = {c: self._segment(c, sol.intervals[c]) for c in members}
        cluster_id = uuid.uuid4().hex[:12]
        needs_review = resid > self.cfg.align.cycle_residual_tol
        res = prune(list(segs.values()), eps=d.eps_full_length, contain_tol=d.contain_tol,
                    rule6_quality_margin=d.rule6_quality_margin,
                    audio_variant_keep_both=d.audio_variant_keep_both,
                    set_cover_pass=d.set_cover_pass, span=sol.canonical_span)
        return ClusterDecision(cluster_id, members, [s.vid for s in res.keep],
                               [s.vid for s in res.drop], res.dominated_by,
                               sol.canonical_span, resid, needs_review, segs)

    def _segment(self, cid, interval) -> Segment:
        q = self.catalog.get_quality(cid) or {}
        am = self.catalog.get_audio_meta(cid)
        lang = ""
        if am and am["tracks_json"]:
            tracks = json.loads(am["tracks_json"])
            dt = am["default_track"] or 0
            if tracks and dt < len(tracks):
                lang = tracks[dt].get("lang_tag") or ""
        if lang.lower() in ("und", "unknown", "none", "mis", "zxx"):
            lang = ""    # unknown/undefined is not a distinct language -> normal dominance
        sm = self.catalog.get_stream_meta(cid)
        codec = (sm["vcodec"] if sm else "") or ""
        row = self.catalog.get_file(cid)
        return Segment(vid=cid, s=interval[0], e=interval[1],
                       quality=q.get("Q_composite", 0.0), terrible=bool(q.get("terrible", False)),
                       audio_quality=q.get("Q_audio", 0.0), lang=lang,
                       codec_rank=_CODEC_RANK.get(codec.lower(), 0),
                       size_bytes=row["size_bytes"] if row else 0)

    # ---- persist ---------------------------------------------------------
    def _persist(self, run_id, clusters, records):
        for (a, b), rec in records.items():
            self.catalog.add_edge(a, b, rec)
        for c in clusters:
            self.catalog.save_cluster(c.cluster_id, c.members, c.canonical_span,
                                      c.solve_residual, c.needs_review)
            for cid in c.members:
                seg = c.segments[cid]
                self.catalog.save_timeline(cid, c.cluster_id, 1.0, seg.s, seg.s, seg.e)
                action = "keep" if cid in c.keep else "prune"
                self.catalog.save_decision(cid, c.cluster_id, run_id, action,
                                           c.dominated_by.get(cid), {"span": c.canonical_span})
        self.catalog.commit()

    def close(self):
        self.catalog.close()
