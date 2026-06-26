"""Benchmark-harness tests: ground-truth parsing, calibration, and recall on
the synthetic fixtures."""
import pytest

from vdedup.eval import parse_ground_truth, evaluate, calibrate_thresholds


def test_parse_ground_truth(tmp_path):
    gt = tmp_path / "pairs.txt"
    gt.write_text("a\\ one.mp4\nb\\ two.mp4\n\nc.mp4\nd.mp4\n")
    pairs = parse_ground_truth(gt)
    assert pairs == [("a one.mp4", "b two.mp4"), ("c.mp4", "d.mp4")]


def test_calibrate_thresholds_uses_low_percentile():
    quals = [{"R_eff": r, "audio_bw_hz": b, "dover_tech": d, "bpp_norm": p}
             for r, b, d, p in [(1e6, 18000, 0.9, 0.1), (8e5, 16000, 0.8, 0.08),
                                (3e5, 9000, 0.5, 0.03), (5e5, 12000, 0.6, 0.05)]]
    sug = calibrate_thresholds(quals, low_pct=5.0)
    # suggested floor sits below the median, above the implausible
    assert sug["terrible_reff_px"] < 8e5
    assert 8000 < sug["terrible_audio_bw_hz"] < 16000
    assert "_stats" in sug


@pytest.mark.media
@pytest.mark.ml
def test_evaluate_recall_on_fixtures(tmp_path, corpus):
    from vdedup.config import Config
    from vdedup.pipeline import Pipeline
    cfg = Config()
    cfg.data_dir = str(tmp_path / "data")
    cfg.root = str(corpus.root)
    cfg.quality.use_vmaf = False
    pipe = Pipeline(cfg)
    if pipe.embedder is None or not pipe.embedder.available:
        pytest.skip("SSCD weights unavailable")
    result = pipe.run()
    # A_full and A_480 are the same title -> must be flagged
    ev = evaluate(result, pipe.catalog, [("A_full.mp4", "A_480.mp4")])
    assert ev.recall == 1.0
    assert len(ev.cross_group_merges) == 0
    pipe.close()
