"""End-to-end pipeline tests against the full synthetic corpus.

Clustering correctness requires SSCD (pHash is too weak to confirm re-encodes,
by design), so those tests are ml-marked. A non-ml smoke test verifies the whole
chain runs and the ingest-level exact-duplicate detection works.
"""
import json
from pathlib import Path

import pytest

from vdedup.config import Config
from vdedup.pipeline import Pipeline
from vdedup.actions import build_plan, apply_plan


def _names(pipe):
    return {row["content_id"]: Path(row["path"]).stem
            for row in pipe.catalog.iter_files()}


def _cid(pipe, name):
    for row in pipe.catalog.iter_files():
        if Path(row["path"]).stem == name:
            return row["content_id"]
    r = pipe.catalog.conn.execute(
        "SELECT content_id FROM dup_path WHERE path LIKE ?", (f"%{name}.%",)).fetchone()
    return r["content_id"] if r else None


def _cluster_of(result, cid):
    for c in result.clusters:
        if cid in c.members:
            return c
    return None


# --------------------------------------------------------------------------
# non-ml smoke test (pHash visual mode): the chain runs, exact-dup detected
# --------------------------------------------------------------------------
@pytest.mark.media
def test_pipeline_runs_and_detects_exact_duplicate(tmp_path, corpus):
    cfg = Config()
    cfg.data_dir = str(tmp_path / "data")
    cfg.root = str(corpus.root)
    cfg.vision.use_sscd = False
    pipe = Pipeline(cfg)
    result = pipe.run()
    assert result.n_files == 10            # 11 files, A_remux dedup'd to A_full
    assert result.n_duplicates >= 1        # the remux
    # every file is accounted for in exactly one cluster
    members = [m for c in result.clusters for m in c.members]
    assert len(members) == len(set(members)) == result.n_files
    pipe.close()


# --------------------------------------------------------------------------
# ml tests (SSCD visual arbiter): full clustering + decision correctness
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sscd_run(tmp_path_factory, corpus):
    cfg = Config()
    cfg.data_dir = str(tmp_path_factory.mktemp("sscd") / "data")
    cfg.root = str(corpus.root)
    cfg.vision.use_sscd = True
    pipe = Pipeline(cfg)
    if pipe.embedder is None or not pipe.embedder.available:
        pytest.skip("SSCD weights unavailable")
    result = pipe.run()
    cids = {name: _cid(pipe, name) for name in
            ["A_full", "A_remux", "A_480", "A_clip_mid", "A_clip_480", "A_letterbox",
             "A_lofi_audio", "A_redub", "A_silent_clip", "B_full", "black_trap"]}
    yield pipe, result, cids
    pipe.close()


pytestmark = pytest.mark.media


@pytest.mark.ml
def test_title_A_forms_one_cluster(sscd_run):
    pipe, result, cids = sscd_run
    a_keys = ["A_full", "A_480", "A_clip_mid", "A_clip_480", "A_letterbox",
              "A_lofi_audio", "A_redub", "A_silent_clip"]
    cluster = _cluster_of(result, cids["A_full"])
    assert cluster is not None
    missing = {cids[k] for k in a_keys} - set(cluster.members)
    assert not missing, f"title-A members not clustered: {missing}"


@pytest.mark.ml
def test_B_and_trap_are_separate_singletons(sscd_run):
    pipe, result, cids = sscd_run
    a_cluster = _cluster_of(result, cids["A_full"])
    assert cids["B_full"] not in a_cluster.members
    assert cids["black_trap"] not in a_cluster.members
    assert len(_cluster_of(result, cids["B_full"]).members) == 1
    assert len(_cluster_of(result, cids["black_trap"]).members) == 1


@pytest.mark.ml
def test_remux_is_exact_duplicate(sscd_run):
    pipe, result, cids = sscd_run
    assert cids["A_remux"] == cids["A_full"]      # remux-invariant content id
    assert result.n_duplicates >= 1


@pytest.mark.ml
def test_full_dominates_clips_and_lower_quality(sscd_run):
    pipe, result, cids = sscd_run
    c = _cluster_of(result, cids["A_full"])
    assert cids["A_full"] in c.keep               # best full-length English kept
    for redundant in ["A_clip_mid", "A_clip_480", "A_silent_clip", "A_480",
                      "A_letterbox", "A_lofi_audio"]:
        assert cids[redundant] in c.drop, f"{redundant} should be pruned"


@pytest.mark.ml
def test_audio_variant_distinct_language_kept(sscd_run):
    pipe, result, cids = sscd_run
    c = _cluster_of(result, cids["A_full"])
    assert cids["A_redub"] in c.keep              # Spanish redub kept alongside English


@pytest.mark.ml
def test_shared_audio_different_video_routed_to_review(sscd_run):
    pipe, result, cids = sscd_run
    # A_redub and B_full share the audio_b soundtrack but are different videos
    pairs = {frozenset((a, b)) for a, b, _ in result.review_pairs}
    assert frozenset((cids["A_redub"], cids["B_full"])) in pairs


@pytest.mark.ml
def test_quarantine_plan_reversible(sscd_run, tmp_path):
    pipe, result, cids = sscd_run
    plan = build_plan(result, pipe.catalog)
    assert len(plan.items) >= 6                   # 6 cluster prunes + the remux dup
    manifest = apply_plan(plan, tmp_path / "q", move=False)   # copy so fixtures survive
    data = json.loads(Path(manifest).read_text())
    assert len(data["items"]) == len(plan.items)
    assert all("original_path" in it for it in data["items"])
