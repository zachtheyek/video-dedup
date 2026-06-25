"""Incrementality: content-addressed feature caches are reused across scans, and
a newly-added file only computes its own features."""
import shutil
from pathlib import Path

import pytest

from vdedup.config import Config
from vdedup.pipeline import Pipeline

pytestmark = pytest.mark.media


def _cfg(tmp_path, root):
    cfg = Config()
    cfg.data_dir = str(tmp_path / "data")
    cfg.root = str(root)
    cfg.vision.use_sscd = False     # caching behaviour is mode-independent
    return cfg


def test_rescan_reuses_caches(tmp_path, corpus):
    cfg = _cfg(tmp_path, corpus.root)
    pipe = Pipeline(cfg)
    r1 = pipe.run()
    pipe.close()

    cache_files = list((Path(cfg.data_dir) / "cache").rglob("*.npz"))
    assert cache_files, "no feature caches written"
    mtimes = {f: f.stat().st_mtime_ns for f in cache_files}

    pipe2 = Pipeline(cfg)
    r2 = pipe2.run()
    pipe2.close()

    # nothing new -> no cache file is rewritten, and the result is stable
    for f in cache_files:
        assert f.stat().st_mtime_ns == mtimes[f], f"{f.name} was recomputed"
    assert r2.n_files == r1.n_files


def test_adding_a_file_only_computes_its_features(tmp_path, corpus):
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy(corpus.files["A_full"], lib / "A_full.mp4")
    shutil.copy(corpus.files["B_full"], lib / "B_full.mp4")

    cfg = _cfg(tmp_path, lib)
    pipe = Pipeline(cfg)
    pipe.run()
    pipe.close()
    cache_files = list((Path(cfg.data_dir) / "cache").rglob("*.npz"))
    mtimes = {f: f.stat().st_mtime_ns for f in cache_files}
    n_before = len(cache_files)

    # add one new file and rescan
    shutil.copy(corpus.files["A_clip_mid"], lib / "A_clip_mid.mp4")
    pipe2 = Pipeline(cfg)
    pipe2.run()
    pipe2.close()

    cache_after = list((Path(cfg.data_dir) / "cache").rglob("*.npz"))
    assert len(cache_after) > n_before                      # new file's caches appeared
    for f, mt in mtimes.items():
        assert f.stat().st_mtime_ns == mt, f"{f.name} recomputed for an unchanged file"
