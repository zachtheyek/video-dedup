#!/usr/bin/env python3
"""Pre-commit guard: block staged content that contains real library filenames or
explicit material, so it can never reach the remote (see the 2026-06 incident).

Two checks on the *staged* content of each text file:
  1. Generic: absolute paths into a user's real media library
     (~/Desktop, ~/Movies, ~/Downloads + a video extension) — the usual leak path
     (a pasted path, or an accidentally-committed catalog dump).
  2. Denylist: any term listed in a local, gitignored `.explicit-denylist`
     (e.g. performer/studio names). The denylist itself is never committed, so
     the explicit terms never enter the repo.

Bypass a false positive with `git commit --no-verify` (and fix .explicit-denylist).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATH_RE = re.compile(
    r"/Users/[^/\s]+/(?:Desktop|Movies|Downloads|Pictures)/[^\s\"']*"
    r"\.(?:mp4|mkv|mov|avi|webm|m4v|flv|ts)", re.I)


def staged_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                         capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def staged_bytes(path: str) -> bytes:
    return subprocess.run(["git", "show", f":{path}"], capture_output=True).stdout


def load_denylist() -> list[str]:
    p = Path(".explicit-denylist")
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def main() -> int:
    deny = load_denylist()
    problems: list[tuple[str, str, str]] = []
    for f in staged_files():
        try:
            text = staged_bytes(f).decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary file
        for m in PATH_RE.finditer(text):
            problems.append((f, "real media path", m.group(0)[:70]))
        low = text.lower()
        for term in deny:
            if term.lower() in low:
                problems.append((f, "denylist term", term))

    if problems:
        sys.stderr.write(
            "\nBLOCKED — staged changes contain real library / explicit content. "
            "Anonymise it (use template names like title_A__release_1) before committing:\n")
        for f, kind, hit in problems:
            sys.stderr.write(f"  {f}: {kind}: {hit}\n")
        sys.stderr.write("(false positive? adjust .explicit-denylist, or `git commit --no-verify`)\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
