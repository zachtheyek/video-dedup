"""Full-reference VMAF (native ffmpeg libvmaf) integration test."""
import subprocess

import pytest

from vdedup.quality.fullref import vmaf, visqol


def _has_libvmaf() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                         capture_output=True, text=True).stdout
    return "libvmaf" in out


pytestmark = [
    pytest.mark.media,
    pytest.mark.skipif(not _has_libvmaf(), reason="this ffmpeg build lacks libvmaf"),
]

M = "tests/fixtures/media/"


def test_vmaf_identical_is_high(corpus):
    v = vmaf(M + "A_full.mp4", M + "A_full.mp4", span=8, ref_w=1280, ref_h=720)
    assert v is not None
    assert v > 95.0


def test_vmaf_degraded_is_lower(corpus):
    ref_self = vmaf(M + "A_full.mp4", M + "A_full.mp4", span=8, ref_w=1280, ref_h=720)
    degraded = vmaf(M + "A_full.mp4", M + "A_480.mp4", span=8, ref_w=1280, ref_h=720)
    assert degraded is not None
    assert degraded < ref_self      # 480p re-encode scores worse than the source


def test_visqol_stub_returns_none():
    assert visqol(M + "A_full.mp4", M + "A_full.mp4") is None
