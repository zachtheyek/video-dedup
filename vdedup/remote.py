"""Remote (adb) video sources.

Process a video library on a USB-connected Android / Meta-Quest device without
copying the whole thing: each file is pulled to a local temp, processed, and
deleted, so the originals never leave the device. Features are cached by
content_id, so an interrupted run resumes without re-pulling anything already
done.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi", ".ts"}


def _adb(args: list[str], serial: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    return subprocess.run(cmd, capture_output=True, text=True)


def devices() -> list[str]:
    out = _adb(["devices"]).stdout
    return [ln.split("\t")[0] for ln in out.splitlines()[1:]
            if ln.strip() and ln.strip().endswith("device")]


def list_videos(remote_dir: str, serial: str | None = None) -> list[str]:
    r = _adb(["shell", f"find '{remote_dir}' -type f 2>/dev/null"], serial)
    out = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return sorted(f for f in out if Path(f).suffix.lower() in VIDEO_EXTS)


def pull(remote_path: str, local_path: str | Path, serial: str | None = None) -> bool:
    r = _adb(["pull", "-a", remote_path, str(local_path)], serial)
    return r.returncode == 0 and Path(local_path).exists()


@dataclass
class FileSpec:
    """A file to ingest. `logical` is what the catalog/report shows; `materialize`
    returns a local path for ffmpeg (pulling if remote) and `cleanup` removes any
    temp it created."""
    logical: str
    remote: bool
    materialize: callable      # () -> local path (or None on failure)
    cleanup: callable          # (local_path) -> None


def local_spec(path: str) -> FileSpec:
    return FileSpec(logical=path, remote=False,
                    materialize=lambda: path, cleanup=lambda _p: None)


def adb_spec(remote_path: str, tmp_dir: Path, serial: str | None = None) -> FileSpec:
    def materialize():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / Path(remote_path).name
        return str(dest) if pull(remote_path, dest, serial) else None

    def cleanup(local):
        try:
            if local:
                Path(local).unlink(missing_ok=True)
        except OSError:
            pass

    return FileSpec(logical=f"adb:{remote_path}", remote=True,
                    materialize=materialize, cleanup=cleanup)
