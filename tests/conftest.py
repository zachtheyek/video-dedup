"""Shared pytest fixtures: build the synthetic corpus once per session."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from make_fixtures import build_corpus  # noqa: E402

MEDIA_DIR = Path(__file__).parent / "fixtures" / "media"


@pytest.fixture(scope="session")
def corpus():
    return build_corpus(MEDIA_DIR)
