"""Stage-3 tests against fixtures: audio fingerprints match the right title at the
right offset; pHash separates same-content from different-content; the entropy
filter rejects black frames."""
import numpy as np
import pytest

from vdedup.config import Config
from vdedup.media import ffmpeg
from vdedup.descriptors import fingerprint, phash_frames, hamming, informative_mask
from vdedup.descriptors.audio_fp import matched_pairs, AudioFingerprint
from vdedup.align import fit_offset, MatchPairs

pytestmark = pytest.mark.media
CFG = Config()


def _fp(path):
    a = CFG.audio
    samples = ffmpeg.decode_audio(path, a.sample_rate)
    return fingerprint(samples, a.sample_rate, frame_size=a.frame_size, hop=a.hop_size,
                       peak_neighborhood=a.peak_neighborhood, fan_value=a.fan_value,
                       max_dt=a.max_hash_time_delta, energy_pct=a.activity_energy_pct,
                       flux_pct=a.activity_flux_pct)


def _pairs_to_mp(ta, tb):
    w = np.ones(len(ta))
    return MatchPairs(ta, tb, w, np.ones(len(ta), dtype=bool))


def test_audio_fingerprint_nonempty(corpus):
    fp = _fp(corpus.files["A_full"])
    assert len(fp) > 200  # a 30s tune yields plenty of landmarks


def test_audio_matches_clip_at_correct_offset(corpus):
    full = _fp(corpus.files["A_full"])
    clip = _fp(corpus.files["A_clip_mid"])   # clip is source[10:22]
    ta, tb = matched_pairs(full, clip)       # (t_full, t_clip)
    assert len(ta) > 30
    fit = fit_offset(_pairs_to_mp(ta, tb), offset_bin=0.25, inlier_tol=0.5)
    # t_clip = t_full - 10  => beta = -10
    assert fit.beta == pytest.approx(-10.0, abs=0.3)
    assert fit.n_inliers > 25


def test_audio_narrowband_still_matches(corpus):
    full = _fp(corpus.files["A_full"])
    lofi = _fp(corpus.files["A_lofi_audio"])  # same tune, low-pass to 5.5 kHz
    ta, tb = matched_pairs(full, lofi)
    assert len(ta) > 20
    fit = fit_offset(_pairs_to_mp(ta, tb), offset_bin=0.25, inlier_tol=0.5)
    assert fit.beta == pytest.approx(0.0, abs=0.3)


def test_audio_different_title_few_matches(corpus):
    full = _fp(corpus.files["A_full"])
    other = _fp(corpus.files["B_full"])       # different melody
    ta, tb = matched_pairs(full, other)
    if len(ta) >= 6:
        fit = fit_offset(_pairs_to_mp(ta, tb), offset_bin=0.25, inlier_tol=0.5)
        # no consistent offset across a real span
        assert not (fit.n_inliers > 15 and fit.span_seconds > 5.0)


def test_audio_redub_does_not_match(corpus):
    full = _fp(corpus.files["A_full"])
    redub = _fp(corpus.files["A_redub"])      # title-A video, title-B soundtrack
    ta, tb = matched_pairs(full, redub)
    if len(ta) >= 6:
        fit = fit_offset(_pairs_to_mp(ta, tb), offset_bin=0.25, inlier_tol=0.5)
        assert not (fit.n_inliers > 15 and fit.span_seconds > 5.0)


def _decode_gray(path, fps=2.0, edge=64):
    pj = ffmpeg.probe(path)
    v = [s for s in pj["streams"] if s["codec_type"] == "video"][0]
    w, h = ffmpeg.fit_long_side(v["width"], v["height"], edge)
    fr, t = ffmpeg.decode_frames(path, fps, w, h, gray=True)
    return fr, t


def test_phash_same_content_low_distance(corpus):
    fa, _ = _decode_gray(corpus.files["A_full"])
    fb, _ = _decode_gray(corpus.files["A_480"])   # same content, 480p re-encode
    n = min(len(fa), len(fb))
    ha = phash_frames(fa[:n])
    hb = phash_frames(fb[:n])
    dists = hamming(ha, hb)
    assert np.median(dists) < 12                  # close hashes for same content


def test_phash_different_content_high_distance(corpus):
    fa, _ = _decode_gray(corpus.files["A_full"])
    fb, _ = _decode_gray(corpus.files["B_full"])
    n = min(len(fa), len(fb))
    dists = hamming(phash_frames(fa[:n]), phash_frames(fb[:n]))
    assert np.median(dists) > 18                  # far hashes for different content


def test_entropy_filter_rejects_black(corpus):
    black, _ = _decode_gray(corpus.files["black_trap"])
    rich, _ = _decode_gray(corpus.files["A_full"])
    assert informative_mask(black, entropy_min=CFG.vision.entropy_min).mean() < 0.1
    assert informative_mask(rich, entropy_min=CFG.vision.entropy_min).mean() > 0.8
