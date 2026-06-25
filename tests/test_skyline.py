"""Decision-engine tests: the six heuristics in isolation, the audio-variant and
silent-file branches, exact-content tiebreaks, and the coverage-preservation
property under randomised fuzzing."""
import random

import pytest

from vdedup.decide import Segment
from vdedup.decide.skyline import prune, covered_fraction


def seg(vid, s, e, q=1.0, terrible=False, aq=0.0, lang="", codec=0, size=0):
    return Segment(vid=vid, s=s, e=e, quality=q, terrible=terrible,
                   audio_quality=aq, lang=lang, codec_rank=codec, size_bytes=size)


# ---- the six heuristics ---------------------------------------------------

def test_h1_terrible_is_last_resort():
    # a terrible file covering a region nothing else covers is KEPT there;
    # but where a non-terrible file overlaps it, the terrible one is dropped.
    full_ok = seg("ok", 0, 100, q=2.0, terrible=False)
    terrible_tail = seg("bad", 90, 120, q=5.0, terrible=True)  # high Q but terrible, extends past 100
    res = prune([full_ok, terrible_tail])
    ids = res.keep_ids
    assert "ok" in ids
    assert "bad" in ids  # survives: covers [100,120] which ok does not

    # now the terrible file is fully inside the good one -> dropped
    terrible_inside = seg("bad2", 10, 90, q=9.0, terrible=True)
    res2 = prune([full_ok, terrible_inside])
    assert res2.keep_ids == {"ok"}
    assert res2.dominated_by["bad2"] == "ok"


def test_h2_prefer_full_over_clip_regardless_of_quality():
    full = seg("full", 0, 100, q=1.0)
    clip = seg("clip", 20, 40, q=9.0)  # higher quality but contained
    res = prune([full, clip])
    assert res.keep_ids == {"full"}


def test_h3_prefer_higher_quality_same_span():
    lo = seg("lo", 0, 50, q=1.0)
    hi = seg("hi", 0, 50, q=2.0)
    res = prune([lo, hi])
    assert res.keep_ids == {"hi"}


def test_h4_complete_overlap_keep_higher_quality():
    a = seg("a", 10, 20, q=1.0)
    b = seg("b", 10, 20, q=3.0)
    res = prune([a, b])
    assert res.keep_ids == {"b"}


def test_h5_partial_overlap_keep_both():
    a = seg("a", 0, 30, q=1.0)
    b = seg("b", 20, 50, q=2.0)  # neither contains the other
    res = prune([a, b])
    assert res.keep_ids == {"a", "b"}


def test_h6_full_low_q_absorbs_higher_q_clips():
    full = seg("full", 0, 100, q=1.0)
    c1 = seg("c1", 0, 30, q=8.0)
    c2 = seg("c2", 40, 80, q=9.0)
    res = prune([full, c1, c2])
    assert res.keep_ids == {"full"}


# ---- policy knobs ---------------------------------------------------------

def test_rule6_delta_override_keeps_much_better_clip():
    full = seg("full", 0, 100, q=1.0)
    clip = seg("clip", 20, 40, q=5.0)
    # default: full absorbs clip
    assert prune([full, clip]).keep_ids == {"full"}
    # with a Δ margin of 2.0 and clip 4.0 better, keep the clip too
    res = prune([full, clip], rule6_quality_margin=2.0)
    assert res.keep_ids == {"full", "clip"}


def test_audio_variant_distinct_language_keeps_both():
    # identical video span, different soundtracks/languages
    en = seg("en", 0, 100, q=2.0, aq=1.0, lang="en")
    fr = seg("fr", 0, 100, q=2.0, aq=0.5, lang="fr")
    res = prune([en, fr])
    assert res.keep_ids == {"en", "fr"}  # distinct language -> keep both


def test_audio_variant_same_language_better_audio_wins():
    a = seg("a", 0, 100, q=2.0, aq=1.0, lang="en")
    b = seg("b", 0, 100, q=2.0, aq=0.3, lang="en")  # same lang, worse audio
    res = prune([a, b])
    assert res.keep_ids == {"a"}


def test_keep_one_per_language():
    en_hi = seg("en_hi", 0, 100, q=2.0, aq=1.0, lang="en")
    en_lo = seg("en_lo", 0, 100, q=2.0, aq=0.2, lang="en")
    fr = seg("fr", 0, 100, q=2.0, aq=0.7, lang="fr")
    res = prune([en_hi, en_lo, fr])
    assert res.keep_ids == {"en_hi", "fr"}


def test_silent_file_loses_only_tiebreak_never_demoted():
    # equal video quality; one silent, one with good audio -> audio breaks tie
    silent = seg("silent", 0, 100, q=2.0, aq=0.0, lang="")
    sound = seg("sound", 0, 100, q=2.0, aq=1.0, lang="en")
    res = prune([silent, sound])
    assert res.keep_ids == {"sound"}

    # but a silent file with clearly better video beats a worse-video sound file
    silent_good = seg("sg", 0, 100, q=5.0, aq=0.0, lang="")
    sound_bad = seg("sb", 0, 100, q=1.0, aq=1.0, lang="en")
    res2 = prune([silent_good, sound_bad])
    assert res2.keep_ids == {"sg"}


def test_exact_content_duplicates_tiebreak_chain():
    # identical interval & video quality; differ in audio, then codec, then size
    a = seg("a", 0, 100, q=2.0, aq=1.0, codec=1, size=1000, lang="en")
    b = seg("b", 0, 100, q=2.0, aq=1.0, codec=2, size=900, lang="en")  # better codec
    res = prune([a, b])
    assert res.keep_ids == {"b"}  # codec_rank breaks the tie

    # equal audio & codec -> smaller file wins
    c = seg("c", 0, 100, q=2.0, aq=1.0, codec=1, size=2000, lang="en")
    d = seg("d", 0, 100, q=2.0, aq=1.0, codec=1, size=500, lang="en")
    res2 = prune([c, d])
    assert res2.keep_ids == {"d"}


def test_set_cover_pass_optional():
    # Design Section 12: a medium-Q clip spanning a region, and two higher-Q
    # clips that jointly cover it, where NONE is full-length w.r.t. the title's
    # canonical span (here the title runs [0,200], so all three are clips).
    span_seg = seg("span", 0, 100, q=2.0)
    left = seg("l", 0, 55, q=3.0)
    right = seg("r", 45, 100, q=3.0)
    title = (0.0, 200.0)
    # default single-element dominance keeps the spanning file (continuity)
    assert prune([span_seg, left, right], span=title).keep_ids == {"span", "l", "r"}
    # with set-cover on, the spanning file is redundant (covered by l+r, both better)
    res = prune([span_seg, left, right], span=title, set_cover_pass=True)
    assert "span" not in res.keep_ids
    assert res.keep_ids == {"l", "r"}


# ---- structural properties ------------------------------------------------

def test_idempotent_and_order_independent():
    cluster = [seg(f"v{i}", random.uniform(0, 50), random.uniform(50, 100),
                   q=random.random(), terrible=random.random() < 0.3)
               for i in range(12)]
    k1 = prune(cluster).keep
    k2 = prune(list(reversed(cluster))).keep
    assert {s.vid for s in k1} == {s.vid for s in k2}
    # applying again is a fixed point
    k3 = prune(k1).keep
    assert {s.vid for s in k1} == {s.vid for s in k3}


def _covers_every_point(kept, cluster, lo, hi, n=400):
    pts = [lo + (hi - lo) * i / n for i in range(n + 1)]
    for p in pts:
        covered_in = any(s.s <= p <= s.e for s in cluster)
        if covered_in:
            covered_out = any(s.s <= p <= s.e for s in kept)
            if not covered_out:
                return False, p
    return True, None


@pytest.mark.parametrize("trial", range(200))
def test_coverage_preservation_fuzz(trial):
    rnd = random.Random(trial)
    n = rnd.randint(1, 10)
    cluster = []
    for i in range(n):
        a = rnd.uniform(0, 90)
        b = a + rnd.uniform(1, 40)
        cluster.append(seg(f"v{i}", a, b, q=rnd.random(),
                           terrible=rnd.random() < 0.4))
    lo, hi = min(s.s for s in cluster), max(s.e for s in cluster)
    # exact containment (tol=0) to exercise the proof without fuzz slack
    res = prune(cluster, contain_tol=0.0)
    ok, p = _covers_every_point(res.keep, cluster, lo, hi)
    assert ok, f"trial {trial}: gap opened at canonical t={p}"


def test_covered_fraction_basic():
    assert covered_fraction(seg("x", 0, 50), 0, 100) == pytest.approx(0.5)
    assert covered_fraction(seg("x", -10, 110), 0, 100) == pytest.approx(1.0)
    assert covered_fraction(seg("x", 200, 300), 0, 100) == 0.0
