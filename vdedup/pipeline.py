"""End-to-end orchestration.

Two-pass by default:
  * Pass 1 (cheap blocking): audio fingerprints (full, fast) + a sparse coarse
    visual signature for every file; group files that plausibly share content.
    Files that match nothing are "distinct" and skip the expensive pass.
  * Pass 2 (precise): dense visual descriptors + quality, only for files in a
    candidate group; then fused alignment, clustering, timeline, decision.

Per-file decode runs in a thread pool (ffmpeg releases the GIL); SSCD inference
is serialized under a lock. Dry-run by default; reversible.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import Config
from .catalog import Catalog, FeatureCache
from .features import FeatureExtractor, FileFeatures
from .ingest.ingest import VIDEO_EXTS
from .ingest.probe import parse_probe
from .ingest.content_id import content_id as compute_content_id
from .media import ffmpeg
from .media.deletterbox import detect_crop
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
    review_pairs: list[tuple[str, str, str]]
    edges: dict[tuple[str, str], dict]
    n_files: int
    n_duplicates: int
    n_active: int = 0
    timings: dict[str, float] = field(default_factory=dict)

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
        self.workers = cfg.workers

    # ---- helpers ----------------------------------------------------------
    def _parallel(self, fn, items, desc):
        """Map fn over items with a progress bar. Sequential (main thread) by
        default — forking ffmpeg from a *worker thread* deadlocks on macOS conda
        Python (see docs/BENCHMARK.md), so threads are strictly opt-in via
        `workers > 1`. The dominant speedups (sparse decode + two-pass) do not
        depend on threads."""
        if not items:
            return []
        if self.workers and self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            out = []
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                for r in tqdm(ex.map(fn, items), total=len(items), desc=desc):
                    out.append(r)
            return out
        return [fn(x) for x in tqdm(items, desc=desc)]

    def _info_crop(self, cid: str):
        row = self.catalog.get_file(cid)
        info = parse_probe(json.loads(row["probe_json"]))
        sm = self.catalog.get_stream_meta(cid)
        crop = json.loads(sm["active_crop"]) if sm and sm["active_crop"] else None
        return row, info, (tuple(crop) if crop else None)

    # ---- public -----------------------------------------------------------
    def run(self, root: str | Path | None = None) -> RunResult:
        import time
        root = root or self.cfg.root
        T = {}

        t = time.time(); n_dups, new_cids = self._scan(root); T["scan"] = time.time() - t
        cids = self.catalog.all_content_ids()
        info_crop = {c: self._info_crop(c) for c in cids}

        t = time.time()
        audio, coarse = self._pass1(cids, info_crop); T["pass1"] = time.time() - t

        if self.cfg.two_pass:
            active = self._candidate_active(cids, audio, coarse, info_crop)
        else:
            active = set(cids)

        t = time.time()
        feats = self._pass2(cids, active, audio, info_crop); T["pass2"] = time.time() - t

        t = time.time()
        edges, records, review = self._match(sorted(active), feats); T["match"] = time.time() - t

        t = time.time()
        clusters = self._cluster_and_decide(cids, edges, records, feats)
        review = self._filter_review(review, clusters); T["decide"] = time.time() - t

        run_id = uuid.uuid4().hex[:12]
        self._persist(run_id, clusters, records)
        return RunResult(run_id, clusters, review, records, len(cids), n_dups,
                         n_active=len(active), timings=T)

    # ---- stage 1: parallel ingest ----------------------------------------
    def _scan(self, root) -> tuple[int, list[str]]:
        paths = [str(p) for p in sorted(Path(root).rglob("*"))
                 if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        todo = []
        for p in paths:
            ex = self.catalog.get_file_by_path(p)
            st = os.stat(p)
            if ex and ex["mtime"] == st.st_mtime and ex["size_bytes"] == st.st_size:
                continue
            todo.append((p, st))

        def compute(item):
            p, st = item
            try:
                pj = ffmpeg.probe(p)
            except ffmpeg.FFmpegError as e:
                return (p, st, None, None, None, str(e))
            info = parse_probe(pj)
            crop = None
            if info.has_video and info.width and info.height and info.duration:
                try:
                    cr = detect_crop(p, info.width, info.height, info.duration,
                                     n_frames=self.cfg.vision.cropdetect_frames)
                    crop = cr.as_tuple() if cr else None
                except ffmpeg.FFmpegError:
                    crop = None
            cid = compute_content_id(p, info.width, info.height, info.duration)
            return (p, st, pj, info, crop, cid)

        results = self._parallel(compute, todo, "ingest") if todo else []
        n_dups = 0
        new_cids = []
        for p, st, pj, info, crop, cid in results:
            if pj is None:
                continue
            if self.catalog.content_id_exists(cid):
                prior = [x for x in self.catalog.get_paths_for_content(cid) if x != p]
                if prior:
                    self.catalog.record_duplicate_path(cid, p)
                    n_dups += 1
                    continue
            self.catalog.upsert_file(cid, p, st.st_size, st.st_mtime, pj)
            self.catalog.set_stream_meta(cid, info.stream_meta(active_crop=crop))
            self.catalog.set_audio_meta(cid, info.has_audio, info.audio_tracks, info.default_track)
            new_cids.append(cid)
        self.catalog.commit()
        return n_dups, new_cids

    # ---- pass 1: cheap audio + coarse visual -----------------------------
    def _pass1(self, cids, info_crop):
        def fn(cid):
            row, info, crop = info_crop[cid]
            af = self.fx.audio(cid, row["path"], info)
            cv = self.fx.coarse_visual(cid, row["path"], info, crop) if self.cfg.two_pass else (
                np.zeros((0, 512), np.float32), np.zeros(0, np.uint64), np.zeros(0))
            return cid, af, cv, info.has_audio
        audio, coarse = {}, {}
        for cid, af, cv, _ha in self._parallel(fn, cids, "pass1 audio+coarse"):
            audio[cid] = af
            coarse[cid] = cv
        return audio, coarse

    def _candidate_active(self, cids, audio, coarse, info_crop) -> set[str]:
        """Recall-oriented blocking: a pair is a candidate if it shares enough
        audio hashes OR enough coarse visual neighbours. Files in a group of >=2
        are 'active' and proceed to dense extraction."""
        mode = self.fx._mode
        aidx = AudioIndex()
        vidx = VisualIndex(mode=mode, sim_min=self.cfg.candidate.visual_sim_min,
                           knn_k=self.cfg.candidate.knn_k, use_faiss=self.cfg.candidate.use_faiss)
        for c in cids:
            aidx.add(c, audio[c])
            v, ph, ct = coarse[c]
            vidx.add(c, v if mode == "embedding" else ph, ct)
        vidx.build()
        cc = self.cfg.candidate
        edges = []
        for c in cids:
            am = aidx.query(c, audio[c])
            for other, lst in am.items():
                if len(lst) >= cc.min_shared_audio_hashes:
                    edges.append(Edge(c, other, 0.0))
            v, ph, ct = coarse[c]
            vm = vidx.query(c, v if mode == "embedding" else ph, ct)
            for other, lst in vm.items():
                if len(lst) >= cc.coarse_visual_min:
                    edges.append(Edge(c, other, 0.0))
        active = set()
        for comp in connected_components(cids, edges):
            if len(comp) >= 2:
                active.update(comp)
        return active

    # ---- pass 2: dense visual + quality (active only) --------------------
    def _pass2(self, cids, active, audio, info_crop):
        # catalog reads must happen on the main thread (SQLite is single-thread);
        # workers only decode + run torch and touch the per-file npz cache.
        need_quality = {c for c in active if self.catalog.get_quality(c) is None}

        def fn(cid):
            row, info, crop = info_crop[cid]
            vecs, ph, vt = self.fx.dense_visual(cid, row["path"], info, crop)
            q = self.fx.score(cid, row["path"], info, crop, info.has_audio) if cid in need_quality else None
            return cid, vecs, ph, vt, q
        results = self._parallel(fn, sorted(active), "pass2 dense+quality") if active else []
        feats = {}
        for cid, vecs, ph, vt, q in results:
            if q is not None:
                self.catalog.set_quality(cid, q.to_dict())
            _row, info, _crop = info_crop[cid]
            feats[cid] = FileFeatures(cid, has_audio=info.has_audio, mode=self.fx._mode,
                                      vecs=vecs, phash=ph, vtimes=vt,
                                      ahashes=audio[cid].hashes, atimes=audio[cid].times,
                                      has_dense=True)
        self.catalog.commit()
        return feats

    # ---- stage 4-5: candidate + verify -----------------------------------
    def _build_indexes(self, cids, feats):
        mode = "embedding" if any(feats[c].mode == "embedding" for c in cids) else "phash"
        aidx = AudioIndex()
        vidx = VisualIndex(mode=mode, sim_min=self.cfg.candidate.visual_sim_min,
                           knn_k=self.cfg.candidate.knn_k, use_faiss=self.cfg.candidate.use_faiss)
        for c in cids:
            f = feats[c]
            aidx.add(c, AudioFingerprint(f.ahashes, f.atimes))
            vidx.add(c, f.vecs if mode == "embedding" else f.phash, f.vtimes)
        vidx.build()
        return aidx, vidx, mode

    def _match(self, cids, feats):
        if not cids:
            return [], {}, []
        aidx, vidx, mode = self._build_indexes(cids, feats)
        al = self.cfg.align
        grid = (np.arange(*self._grid()).tolist() if al.scale_search else (1.0,))
        edges, records, review = [], {}, []
        processed = set()
        for a in cids:
            f = feats[a]
            am = aidx.query(a, AudioFingerprint(f.ahashes, f.atimes), self.cfg.candidate.idf_smoothing)
            vm = vidx.query(a, f.vecs if mode == "embedding" else f.phash, f.vtimes)
            for cand in generate_candidates(a, am, vm, self.cfg):
                b = cand.other
                key = frozenset((a, b))
                if key in processed or b not in feats:
                    continue
                processed.add(key)
                dec = verify_pair(cand.pairs, has_audio_a=f.has_audio,
                                  has_audio_b=feats[b].has_audio, cfg=al, alpha_grid=grid)
                records[(a, b)] = dict(
                    alpha=dec.alpha, beta=dec.beta, v_inliers=dec.v_inliers, a_inliers=dec.a_inliers,
                    span_seconds=dec.span_seconds, residual_std=dec.residual_std, modality=dec.modality,
                    audio_agrees=dec.audio_agrees, confidence=dec.confidence,
                    route_review=dec.route_review, piecewise=dec.piecewise, reason=dec.reason)
                if dec.accept and dec.modality in _VIDEO_GROUNDED:
                    w = dec.a_inliers / al.audio_offset_var + dec.v_inliers / al.visual_offset_var
                    edges.append(Edge(a, b, dec.beta, max(w, 1e-6)))
                    if dec.route_review:
                        review.append((a, b, f"{dec.modality}: {dec.reason}"))
                elif dec.modality == "audio" or dec.route_review:
                    review.append((a, b, dec.reason))
        return edges, records, review

    def _grid(self):
        lo, hi, step = self.cfg.align.alpha_grid
        return (lo, hi + step / 2, step)

    # ---- stage 6-7-9 ------------------------------------------------------
    def _cluster_and_decide(self, cids, edges, records, feats) -> list[ClusterDecision]:
        out = []
        for comp in connected_components(cids, edges):
            comp_edges = [e for e in edges if e.i in comp and e.j in comp]
            for sub, sub_edges, resid in self._solve_with_cycle_check(comp, comp_edges):
                out.append(self._decide_cluster(sub, sub_edges, resid))
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
        if len(members) > 1 and self.cfg.quality.use_vmaf:
            self._vmaf_refine(members, segs, sol)
        cluster_id = uuid.uuid4().hex[:12]
        needs_review = resid > self.cfg.align.cycle_residual_tol
        res = prune(list(segs.values()), eps=d.eps_full_length, contain_tol=d.contain_tol,
                    rule6_quality_margin=d.rule6_quality_margin,
                    audio_variant_keep_both=d.audio_variant_keep_both,
                    set_cover_pass=d.set_cover_pass, span=sol.canonical_span)
        return ClusterDecision(cluster_id, members, [s.vid for s in res.keep],
                               [s.vid for s in res.drop], res.dominated_by,
                               sol.canonical_span, resid, needs_review, segs)

    def _vmaf_refine(self, members, segs, sol):
        """Alignment-relative VMAF: rank members against the highest-R_eff,
        non-terrible pseudo-reference over their overlapping span. Nudges the
        quality scalar so the relative ordering reflects true picture fidelity."""
        from .quality.fullref import vmaf
        q = {c: (self.catalog.get_quality(c) or {}) for c in members}
        cand = [c for c in members if not q[c].get("terrible")]
        if len(cand) < 2:
            return
        ref = max(cand, key=lambda c: q[c].get("R_eff", 0.0))
        rrow = self.catalog.get_stream_meta(ref)
        rw, rh = (rrow["declared_w"], rrow["declared_h"]) if rrow else (None, None)
        ref_path = (self.catalog.get_file(ref) or {})["path"]
        for c in members:
            if c == ref:
                continue
            a, b = segs[ref], segs[c]
            lo, hi = max(a.s, b.s), min(a.e, b.e)
            if hi - lo < 5.0:
                continue
            crow = self.catalog.get_file(c)
            v = vmaf(ref_path, crow["path"], ref_offset=lo - a.s, dist_offset=lo - b.s,
                     span=min(60.0, hi - lo), ref_w=rw, ref_h=rh)
            if v is not None:
                # small monotone nudge in [0, 0.1] so VMAF refines but never
                # overturns the coarse NR ordering
                seg = segs[c]
                segs[c] = Segment(seg.vid, seg.s, seg.e, seg.quality * (0.9 + 0.1 * v / 100.0),
                                  seg.terrible, seg.audio_quality, seg.lang, seg.codec_rank, seg.size_bytes)

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
            lang = ""
        if not lang and self.cfg.decide.use_lid:
            from . import lid
            if lid.available():
                row0 = self.catalog.get_file(cid)
                samples = ffmpeg.decode_audio(row0["path"], 16000)[: 16000 * 8]
                lang = lid.identify(samples, 16000) or ""
        sm = self.catalog.get_stream_meta(cid)
        codec = (sm["vcodec"] if sm else "") or ""
        row = self.catalog.get_file(cid)
        return Segment(vid=cid, s=interval[0], e=interval[1],
                       quality=q.get("Q_composite", 0.0), terrible=bool(q.get("terrible", False)),
                       audio_quality=q.get("Q_audio", 0.0), lang=lang,
                       codec_rank=_CODEC_RANK.get(codec.lower(), 0),
                       size_bytes=row["size_bytes"] if row else 0)

    @staticmethod
    def _filter_review(review, clusters):
        member_of = {}
        for c in clusters:
            for cid in c.members:
                member_of[cid] = c.cluster_id
        out, seen = [], set()
        for a, b, reason in review:
            if member_of.get(a) is not None and member_of.get(a) == member_of.get(b):
                continue
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b, reason))
        return out

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
