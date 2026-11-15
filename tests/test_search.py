"""Smoke tests for Phase 0.

Only test_config_defaults should pass right now.
Other tests are marked skip until Phase 2.
"""
import pytest

from src.config import Config


def test_config_defaults():
    """Config should have sensible non-empty defaults."""
    cfg = Config()
    assert cfg.embedding_dim > 0
    assert cfg.top_k > 0
    assert cfg.model_name


@pytest.mark.skip(reason="Implemented in Phase 2")
def test_embed_text_shape():
    """embed_text() should return a normalized (embedding_dim,) vector."""
    pass


@pytest.mark.skip(reason="Implemented in Phase 2")
def test_build_and_search():
    """build_index() + search() should return the correct image in top-1."""
    pass