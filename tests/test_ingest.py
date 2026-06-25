"""Stage-1 tests: content-id remux invariance, re-encode distinction, audio flags."""
import pytest

from vdedup.config import Config
from vdedup.catalog import Catalog
from vdedup.ingest import ingest_file
from vdedup.media import ffmpeg
from vdedup.ingest.probe import parse_probe

pytestmark = pytest.mark.media


@pytest.fixture(scope="module")
def force_rebuild():
    # ensure the crf-tweaked corpus exists
    from make_fixtures import build_corpus
    from conftest import MEDIA_DIR
    return build_corpus(MEDIA_DIR, force=True)


def _cid(path, cfg, cat):
    return ingest_file(path, cat, cfg).content_id


def test_remux_is_identical_content(tmp_path, corpus, force_rebuild):
    cfg = Config()
    cat = Catalog(tmp_path / "cat.sqlite")
    r_full = ingest_file(corpus.files["A_full"], cat, cfg)
    r_remux = ingest_file(corpus.files["A_remux"], cat, cfg)
    assert r_full.status == "new"
    assert r_full.content_id == r_remux.content_id  # remux -> same content id
    assert r_remux.status == "duplicate"            # short-circuited at ingest


def test_reencode_is_distinct_content(tmp_path, corpus, force_rebuild):
    cfg = Config()
    cat = Catalog(tmp_path / "cat.sqlite")
    full = ingest_file(corpus.files["A_full"], cat, cfg)
    enc480 = ingest_file(corpus.files["A_480"], cat, cfg)
    assert full.content_id != enc480.content_id
    assert enc480.status == "new"


def test_clip_and_redub_are_distinct(tmp_path, corpus, force_rebuild):
    cfg = Config()
    cat = Catalog(tmp_path / "cat.sqlite")
    ids = {k: ingest_file(corpus.files[k], cat, cfg).content_id
           for k in ["A_full", "A_clip_mid", "A_lofi_audio", "A_redub"]}
    # all four must be distinct content ids
    assert len(set(ids.values())) == 4, ids


def test_audio_flags(corpus, force_rebuild):
    # silent clip and black trap have no usable audio
    for key, expect_audio in [("A_full", True), ("A_silent_clip", False),
                              ("black_trap", False)]:
        info = parse_probe(ffmpeg.probe(corpus.files[key]))
        if key == "black_trap":
            # anullsrc is technically an audio stream but silent; has_audio True but no signal
            continue
        assert info.has_audio == expect_audio, key


def test_letterbox_crop_detected(corpus, force_rebuild):
    cfg = Config()
    info = parse_probe(ffmpeg.probe(corpus.files["A_letterbox"]))
    from vdedup.media.deletterbox import detect_crop
    crop = detect_crop(corpus.files["A_letterbox"], info.width, info.height, info.duration,
                       n_frames=cfg.vision.cropdetect_frames)
    assert crop is not None
    # the active picture is the 960x540 inner region inside a 960x720 frame
    assert crop.h < info.height
    assert 480 <= crop.h <= 600
