"""SSCD embedding tests (ml-marked: needs torch + downloaded weights).

Run with:  pytest -m ml   (skips automatically if weights can't be fetched)."""
import numpy as np
import pytest

from vdedup.media import ffmpeg
from vdedup.descriptors.embed import Embedder

pytestmark = [pytest.mark.media, pytest.mark.ml]


@pytest.fixture(scope="module")
def embedder():
    e = Embedder(models_dir="models")
    if not e.available:
        pytest.skip("SSCD weights unavailable (no torch or download failed)")
    return e


def _emb(e, path, fps=1.0):
    pj = ffmpeg.probe(path)
    v = [s for s in pj["streams"] if s["codec_type"] == "video"][0]
    w, h = ffmpeg.fit_long_side(v["width"], v["height"], 360)
    fr, t = ffmpeg.decode_frames(path, fps, w, h, gray=False)
    return e.embed(fr)


def test_sscd_output_normalised(embedder):
    frames = (np.random.rand(3, 120, 120, 3) * 255).astype("uint8")
    emb = embedder.embed(frames)
    assert emb.shape == (3, 512)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


def test_sscd_copy_detection_separates_content(embedder, corpus):
    ea = _emb(embedder, corpus.files["A_full"])
    eb = _emb(embedder, corpus.files["A_480"])    # same content, re-encoded 480p
    ec = _emb(embedder, corpus.files["B_full"])   # different title
    n = min(len(ea), len(eb), len(ec))
    same = np.median(np.sum(ea[:n] * eb[:n], axis=1))
    diff = np.median(np.sum(ea[:n] * ec[:n], axis=1))
    assert same > 0.8          # copy survives re-encode + rescale
    assert diff < 0.6          # different content is well separated
    assert same - diff > 0.3
