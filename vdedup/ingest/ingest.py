"""Stage 1 — ingest and inventory.

Enumerate the tree, probe new/changed files, compute content_id, and short-circuit
exact-content duplicates (same decoded content, different container/path) straight
to the decision queue without re-processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..media import ffmpeg
from ..media.deletterbox import detect_crop
from .probe import parse_probe, StreamInfo
from .content_id import content_id

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".ts"}


@dataclass
class IngestResult:
    path: str
    content_id: str
    status: str           # "new" | "duplicate" | "skipped" | "error"
    info: StreamInfo | None = None
    crop: tuple[int, int, int, int] | None = None
    error: str | None = None


def ingest_file(path: str | Path, catalog, cfg, *, deletterbox: bool = True) -> IngestResult:
    path = str(path)
    try:
        st = Path(path).stat()
    except OSError as e:
        return IngestResult(path, "", "error", error=str(e))

    existing = catalog.get_file_by_path(path)
    if existing and existing["mtime"] == st.st_mtime and existing["size_bytes"] == st.st_size:
        return IngestResult(path, existing["content_id"], "skipped")

    try:
        pj = ffmpeg.probe(path)
    except ffmpeg.FFmpegError as e:
        return IngestResult(path, "", "error", error=str(e))
    info = parse_probe(pj)

    crop = None
    if deletterbox and info.has_video and info.width and info.height and info.duration:
        try:
            cr = detect_crop(path, info.width, info.height, info.duration,
                             n_frames=cfg.vision.cropdetect_frames)
            crop = cr.as_tuple() if cr else None
        except ffmpeg.FFmpegError:
            crop = None

    cid = content_id(path, info.width, info.height, info.duration)

    if catalog.content_id_exists(cid):
        prior = [p for p in catalog.get_paths_for_content(cid) if p != path]
        if prior:
            catalog.record_duplicate_path(cid, path)
            return IngestResult(path, cid, "duplicate", info=info, crop=crop)

    catalog.upsert_file(cid, path, st.st_size, st.st_mtime, pj)
    catalog.set_stream_meta(cid, info.stream_meta(active_crop=crop))
    catalog.set_audio_meta(cid, info.has_audio, info.audio_tracks, info.default_track)
    return IngestResult(path, cid, "new", info=info, crop=crop)


def scan_tree(root: str | Path, catalog, cfg, *, deletterbox: bool = True):
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield ingest_file(p, catalog, cfg, deletterbox=deletterbox)
