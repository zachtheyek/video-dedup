"""Fused offset estimation + verification/decision-table tests."""
import numpy as np
import pytest

from vdedup.align import fit_offset, MatchPairs, verify_pair
from vdedup.config import AlignCfg


def make_pairs(specs, seed=0):
    """specs: list of dicts {offset, n, audio, lo, hi, alpha, noise, weight, jitter}.
    'jitter' adds uniform outlier pairs (no consistent offset)."""
    rng = np.random.default_rng(seed)
    ta, tb, w, isa = [], [], [], []
    for sp in specs:
        n = sp["n"]
        a = sp.get("alpha", 1.0)
        if sp.get("jitter"):
            t_a = rng.uniform(sp.get("lo", 0), sp.get("hi", 60), n)
            t_b = rng.uniform(sp.get("lo", 0), sp.get("hi", 60), n)
        else:
            t_a = rng.uniform(sp.get("lo", 0), sp.get("hi", 60), n)
            t_b = a * t_a + sp["offset"] + rng.normal(0, sp.get("noise", 0.02), n)
        ta.append(t_a); tb.append(t_b)
        w.append(np.full(n, sp.get("weight", 1.0)))
        isa.append(np.full(n, sp.get("audio", False), dtype=bool))
    if not ta:
        return MatchPairs.empty()
    return MatchPairs(np.concatenate(ta), np.concatenate(tb),
                      np.concatenate(w), np.concatenate(isa))


CFG = AlignCfg()
GRID = np.arange(0.96, 1.041, 0.005)


def test_recovers_known_offset():
    pairs = make_pairs([{"offset": 12.5, "n": 50, "audio": True}])
    fit = fit_offset(pairs, offset_bin=CFG.offset_bin, inlier_tol=CFG.inlier_tol)
    assert fit.beta == pytest.approx(12.5, abs=0.1)
    assert fit.alpha == pytest.approx(1.0, abs=0.01)
    assert fit.n_inliers >= 45


def test_robust_to_outliers():
    # 40 true audio pairs at offset 7.0, plus 60 random outliers
    pairs = make_pairs([{"offset": 7.0, "n": 40, "audio": True},
                        {"jitter": True, "n": 60, "audio": True}], seed=1)
    fit = fit_offset(pairs, offset_bin=CFG.offset_bin, inlier_tol=CFG.inlier_tol)
    assert fit.beta == pytest.approx(7.0, abs=0.2)
    assert fit.n_inliers >= 35


def test_recovers_time_scale_pal_ntsc():
    pairs = make_pairs([{"offset": 3.0, "alpha": 1.04, "n": 80, "audio": True}], seed=2)
    fit = fit_offset(pairs, offset_bin=CFG.offset_bin, alpha_grid=GRID,
                     inlier_tol=CFG.inlier_tol)
    assert fit.alpha == pytest.approx(1.04, abs=0.01)
    assert fit.beta == pytest.approx(3.0, abs=0.3)


def test_single_coincidental_match_rejected_by_span():
    # many inliers but all at ~one instant (a shared stock shot / stinger)
    pairs = make_pairs([{"offset": 4.0, "n": 30, "audio": False, "lo": 5.0, "hi": 5.05}])
    dec = verify_pair(pairs, has_audio_a=False, has_audio_b=False, cfg=CFG)
    assert dec.modality == "reject"
    assert not dec.accept


def test_piecewise_detection():
    # two consistent offsets over disjoint spans (director's cut style)
    pairs = make_pairs([{"offset": 0.0, "n": 40, "audio": True, "lo": 0, "hi": 30},
                        {"offset": 20.0, "n": 40, "audio": True, "lo": 40, "hi": 70}], seed=3)
    fit = fit_offset(pairs, offset_bin=CFG.offset_bin, inlier_tol=CFG.inlier_tol,
                     piecewise_min_segment=5.0)
    assert fit.piecewise is not None
    assert len(fit.piecewise) == 2


# ---- modality decision table ----------------------------------------------

def test_table_both_agree():
    pairs = make_pairs([{"offset": 9.0, "n": 40, "audio": True},
                        {"offset": 9.0, "n": 30, "audio": False}], seed=4)
    dec = verify_pair(pairs, has_audio_a=True, has_audio_b=True, cfg=CFG)
    assert dec.modality == "both"
    assert dec.accept and dec.audio_agrees
    assert dec.beta == pytest.approx(9.0, abs=0.2)


def test_table_vision_only_silent():
    # good vision, no audio pairs, at least one file silent -> visual, no penalty
    pairs = make_pairs([{"offset": 5.0, "n": 40, "audio": False}], seed=5)
    dec = verify_pair(pairs, has_audio_a=False, has_audio_b=True, cfg=CFG)
    assert dec.modality == "visual"
    assert dec.accept


def test_table_audio_variant_replaced_soundtrack():
    # good vision, both files HAVE audio, but no shared audio anchors -> redub/mute
    pairs = make_pairs([{"offset": 5.0, "n": 40, "audio": False}], seed=6)
    dec = verify_pair(pairs, has_audio_a=True, has_audio_b=True, cfg=CFG)
    assert dec.modality == "audio_variant"
    assert dec.accept and not dec.audio_agrees


def test_table_audio_only_routes_to_review():
    # good audio, vision is just noise -> commentary/reaction; do not merge
    pairs = make_pairs([{"offset": 8.0, "n": 40, "audio": True},
                        {"jitter": True, "n": 40, "audio": False}], seed=7)
    dec = verify_pair(pairs, has_audio_a=True, has_audio_b=True, cfg=CFG)
    assert dec.modality == "audio"
    assert not dec.accept and dec.route_review


def test_table_reject_coincidental():
    pairs = make_pairs([{"jitter": True, "n": 20, "audio": True},
                        {"jitter": True, "n": 20, "audio": False}], seed=8)
    dec = verify_pair(pairs, has_audio_a=True, has_audio_b=True, cfg=CFG)
    assert dec.modality == "reject"
    assert not dec.accept
