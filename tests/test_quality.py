"""Stage-8 tests: NR quality ladders and the TERRIBLE gate against fixtures."""
import numpy as np
import pytest

from vdedup.config import Config
from vdedup.media import ffmpeg
from vdedup.ingest.probe import parse_probe
from vdedup.quality import score_file, audio_bandwidth, r_eff

pytestmark = pytest.mark.media
CFG = Config()
QUALITY_SR = 44100   # quality audio is decoded at the original rate to see codec cutoffs


def _meta(path):
    info = parse_probe(ffmpeg.probe(path))
    m = info.stream_meta()
    m["audio_tracks"] = info.audio_tracks
    return m, info


def _gray_native(path, fps=1.0):
    info = parse_probe(ffmpeg.probe(path))
    w, h = ffmpeg.fit_long_side(info.width, info.height, None)
    fr, _ = ffmpeg.decode_frames(path, fps, w, h, gray=True)
    return fr


def _score(path):
    meta, info = _meta(path)
    gray = _gray_native(path)
    audio = ffmpeg.decode_audio(path, QUALITY_SR) if info.has_audio else np.zeros(0, np.float32)
    return score_file(meta, gray, audio, QUALITY_SR, info.has_audio, CFG)


def test_reff_resolution_ladder(corpus):
    g_full = _gray_native(corpus.files["A_full"])    # 720p
    g_480 = _gray_native(corpus.files["A_480"])       # 480p re-encode
    reff_full, _ = r_eff(1280, 720, g_full)
    reff_480, _ = r_eff(854, 480, g_480)
    assert reff_full > reff_480


def test_video_quality_ladder(corpus):
    q_full = _score(corpus.files["A_full"])
    q_480 = _score(corpus.files["A_480"])
    assert q_full.Q_video > q_480.Q_video


def test_audio_bandwidth_ladder(corpus):
    a_full = ffmpeg.decode_audio(corpus.files["A_full"], QUALITY_SR)
    a_lofi = ffmpeg.decode_audio(corpus.files["A_lofi_audio"], QUALITY_SR)
    bw_full = audio_bandwidth(a_full, QUALITY_SR)
    bw_lofi = audio_bandwidth(a_lofi, QUALITY_SR)
    assert bw_full > bw_lofi
    assert bw_lofi < 7500          # the 5.5 kHz brick-wall is detected
    assert bw_full > 9000          # the clean track retains high-frequency content


def test_terrible_gate_trips_on_lowfi_audio(corpus):
    q = _score(corpus.files["A_lofi_audio"])
    assert q.terrible
    assert "audio_bw" in q.terrible_reason


def test_silent_file_not_gated_for_silence(corpus):
    q = _score(corpus.files["A_silent_clip"])    # 720p, no audio
    assert not q.terrible                          # good video, silence is not bad audio
    assert q.Q_audio == 0.0


def test_clean_full_not_terrible(corpus):
    q = _score(corpus.files["A_full"])
    assert not q.terrible
    assert q.Q_composite > 0
