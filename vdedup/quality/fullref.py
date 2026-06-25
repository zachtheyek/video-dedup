"""Stage 8 — alignment-enabled full-reference cross-check.

Because cluster members are temporally aligned, relative quality can be measured
directly — more reliable than absolute NR scores. VMAF (video) runs via native
ffmpeg `libvmaf`; each member is compared against the cluster's pseudo-reference
(highest R_eff, non-terrible) over their overlapping canonical span, the
distorted input scaled to the reference resolution. ViSQOL (audio) is stubbed
behind the same interface (no pip wheel; Bazel/C++ build).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def vmaf(reference: str | Path, distorted: str | Path, *,
         ref_offset: float = 0.0, dist_offset: float = 0.0,
         span: float | None = None, ref_w: int | None = None, ref_h: int | None = None
         ) -> float | None:
    """Pooled-mean VMAF of `distorted` vs `reference` over their overlapping
    canonical span. Offsets are each file's local start time that maps to the
    same canonical instant. Returns None on any failure (caller falls back to NR).
    """
    scale = f"scale={ref_w}:{ref_h}:flags=bicubic," if ref_w and ref_h else ""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "vmaf.json"
        ss_ref = ["-ss", str(ref_offset)] if ref_offset else []
        ss_dist = ["-ss", str(dist_offset)] if dist_offset else []
        t = ["-t", str(span)] if span else []
        # distorted is input 0, reference is input 1
        lavfi = (f"[0:v]{scale}setpts=PTS-STARTPTS,fps=24[dist];"
                 f"[1:v]scale={ref_w}:{ref_h}:flags=bicubic,setpts=PTS-STARTPTS,fps=24[ref];"
                 if ref_w and ref_h else
                 "[0:v]setpts=PTS-STARTPTS,fps=24[dist];[1:v]setpts=PTS-STARTPTS,fps=24[ref];")
        lavfi += f"[dist][ref]libvmaf=log_fmt=json:log_path={log}"
        cmd = (["ffmpeg", "-v", "error", *ss_dist, *t, "-i", str(distorted),
                *ss_ref, *t, "-i", str(reference), "-lavfi", lavfi, "-f", "null", "-"])
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if p.returncode != 0 or not log.exists():
            return None
        try:
            data = json.loads(log.read_text())
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception:
            return None


def visqol(reference, distorted, **kwargs) -> float | None:
    """Stub — ViSQOL has no pip wheel; the audio NR metrics cover the gate.
    Wire by shelling out to a local `visqol` binary here if one is installed."""
    return None
