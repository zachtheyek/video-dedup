"""Stage-spanning catalog (SQLite).

Single-file, transactional, trivially backed up. Holds metadata and decisions;
large per-file features (frame vectors, audio hashes) are cached on disk by
`FeatureCache`, keyed by content_id, not stuffed into rows. Schema mirrors the
design's Section-3 tables, with JSON columns where a fixed schema would only add
friction.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS file (
    content_id   TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    size_bytes   INTEGER,
    mtime        REAL,
    probe_json   TEXT,
    ingested_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_file_path ON file(path);

CREATE TABLE IF NOT EXISTS stream_meta (
    content_id   TEXT PRIMARY KEY,
    declared_w   INTEGER, declared_h INTEGER, declared_fps REAL,
    declared_bitrate INTEGER, vcodec TEXT, pix_fmt TEXT,
    declared_duration REAL, active_crop TEXT, vfr INTEGER
);

CREATE TABLE IF NOT EXISTS audio_meta (
    content_id   TEXT PRIMARY KEY,
    has_audio    INTEGER, n_tracks INTEGER, tracks_json TEXT, default_track INTEGER
);

CREATE TABLE IF NOT EXISTS quality (
    content_id   TEXT PRIMARY KEY,
    q_json       TEXT,           -- full component breakdown
    q_composite  REAL, q_video REAL, q_audio REAL,
    terrible     INTEGER, terrible_reason TEXT
);

CREATE TABLE IF NOT EXISTS match_edge (
    a TEXT, b TEXT,
    alpha REAL, beta REAL, v_inliers INTEGER, a_inliers INTEGER,
    span_seconds REAL, residual_std REAL, modality TEXT,
    audio_agrees INTEGER, confidence REAL, route_review INTEGER, piecewise INTEGER,
    PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS cluster (
    cluster_id   TEXT PRIMARY KEY,
    members_json TEXT, canonical_span TEXT, solve_residual REAL, needs_review INTEGER
);

CREATE TABLE IF NOT EXISTS timeline (
    content_id TEXT, cluster_id TEXT,
    a_i REAL, b_i REAL, s_canonical REAL, e_canonical REAL,
    PRIMARY KEY (content_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS decision (
    content_id TEXT, cluster_id TEXT, run_id TEXT,
    action TEXT, dominated_by TEXT, evidence_json TEXT,
    PRIMARY KEY (content_id, cluster_id, run_id)
);
"""


class Catalog:
    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.conn.commit()
        self.close()

    # ---- file / inventory -------------------------------------------------
    def content_id_exists(self, content_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM file WHERE content_id=?", (content_id,))
        return cur.fetchone() is not None

    def get_file_by_path(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM file WHERE path=?", (path,)).fetchone()

    def get_file(self, content_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM file WHERE content_id=?", (content_id,)).fetchone()

    def get_stream_meta(self, content_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM stream_meta WHERE content_id=?", (content_id,)).fetchone()

    def get_paths_for_content(self, content_id: str) -> list[str]:
        # tracks all known paths via the duplicate side-table
        rows = self.conn.execute(
            "SELECT path FROM file WHERE content_id=? UNION SELECT path FROM dup_path WHERE content_id=?",
            (content_id, content_id)).fetchall() if self._has_dup_table() else \
            self.conn.execute("SELECT path FROM file WHERE content_id=?", (content_id,)).fetchall()
        return [r["path"] for r in rows]

    def _has_dup_table(self) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dup_path'").fetchone() is not None

    def upsert_file(self, content_id: str, path: str, size_bytes: int, mtime: float,
                    probe: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO file(content_id,path,size_bytes,mtime,probe_json,ingested_at)"
            " VALUES(?,?,?,?,?,?)",
            (content_id, path, size_bytes, mtime, json.dumps(probe), time.time()))

    def record_duplicate_path(self, content_id: str, path: str) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS dup_path (content_id TEXT, path TEXT, PRIMARY KEY(content_id,path))")
        self.conn.execute("INSERT OR IGNORE INTO dup_path(content_id,path) VALUES(?,?)",
                          (content_id, path))

    def set_stream_meta(self, content_id: str, m: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO stream_meta(content_id,declared_w,declared_h,declared_fps,"
            "declared_bitrate,vcodec,pix_fmt,declared_duration,active_crop,vfr) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (content_id, m.get("declared_w"), m.get("declared_h"), m.get("declared_fps"),
             m.get("declared_bitrate"), m.get("vcodec"), m.get("pix_fmt"),
             m.get("declared_duration"), json.dumps(m.get("active_crop")), int(m.get("vfr", False))))

    def set_audio_meta(self, content_id: str, has_audio: bool, tracks: list[dict],
                       default_track: int | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO audio_meta(content_id,has_audio,n_tracks,tracks_json,default_track)"
            " VALUES(?,?,?,?,?)",
            (content_id, int(has_audio), len(tracks), json.dumps(tracks), default_track))

    def get_audio_meta(self, content_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM audio_meta WHERE content_id=?", (content_id,)).fetchone()

    def iter_files(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute("SELECT * FROM file")

    def all_content_ids(self) -> list[str]:
        return [r["content_id"] for r in self.conn.execute("SELECT content_id FROM file")]

    # ---- quality ----------------------------------------------------------
    def set_quality(self, content_id: str, q: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO quality(content_id,q_json,q_composite,q_video,q_audio,terrible,terrible_reason)"
            " VALUES(?,?,?,?,?,?,?)",
            (content_id, json.dumps(q), q.get("Q_composite"), q.get("Q_video"), q.get("Q_audio"),
             int(q.get("terrible", False)), q.get("terrible_reason")))

    def get_quality(self, content_id: str) -> dict | None:
        row = self.conn.execute("SELECT q_json FROM quality WHERE content_id=?", (content_id,)).fetchone()
        return json.loads(row["q_json"]) if row else None

    # ---- edges / clusters / decisions ------------------------------------
    def add_edge(self, a: str, b: str, e: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO match_edge(a,b,alpha,beta,v_inliers,a_inliers,span_seconds,"
            "residual_std,modality,audio_agrees,confidence,route_review,piecewise)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (a, b, e["alpha"], e["beta"], e["v_inliers"], e["a_inliers"], e["span_seconds"],
             e["residual_std"], e["modality"], int(e["audio_agrees"]), e["confidence"],
             int(e.get("route_review", False)), int(e.get("piecewise", False))))

    def save_cluster(self, cluster_id: str, members: list[str], span: tuple[float, float],
                     residual: float, needs_review: bool) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cluster(cluster_id,members_json,canonical_span,solve_residual,needs_review)"
            " VALUES(?,?,?,?,?)",
            (cluster_id, json.dumps(members), json.dumps(span), residual, int(needs_review)))

    def save_timeline(self, content_id: str, cluster_id: str, a_i: float, b_i: float,
                      s: float, e: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO timeline(content_id,cluster_id,a_i,b_i,s_canonical,e_canonical)"
            " VALUES(?,?,?,?,?,?)", (content_id, cluster_id, a_i, b_i, s, e))

    def save_decision(self, content_id: str, cluster_id: str, run_id: str, action: str,
                      dominated_by: str | None, evidence: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO decision(content_id,cluster_id,run_id,action,dominated_by,evidence_json)"
            " VALUES(?,?,?,?,?,?)",
            (content_id, cluster_id, run_id, action, dominated_by, json.dumps(evidence)))

    def commit(self) -> None:
        self.conn.commit()
