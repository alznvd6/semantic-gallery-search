"""Smoke tests for the config and model loader.

Search and indexing tests are marked skip until Phase 2.
"""
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import Config
from src.search import _load_model, embed_text, load_model


def test_config_defaults():
    """Config should have sensible non-empty defaults."""
    cfg = Config()
    assert cfg.embedding_dim > 0
    assert cfg.top_k > 0
    assert cfg.model_name == "jinaai/jina-clip-v2"


def test_load_model(monkeypatch, request):
    """Load Jina CLIP with remote code enabled and prepare it for inference."""
    _load_model.cache_clear()
    request.addfinalizer(_load_model.cache_clear)
    calls = {}

    class FakeModel:
        def to(self, device):
            calls["device"] = device

        def eval(self):
            calls["eval"] = True

    model = FakeModel()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, trust_remote_code):
            calls["model_name"] = model_name
            calls["trust_remote_code"] = trust_remote_code
            return model

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel)
    )
    cfg = Config(model_name="test/jina-clip-v2", device="cpu")

    assert load_model(cfg) is model
    assert calls == {
        "model_name": "test/jina-clip-v2",
        "trust_remote_code": True,
        "device": "cpu",
        "eval": True,
    }


def test_embed_text_shape(monkeypatch):
    """Encode retrieval queries into normalized configured-size vectors."""
    calls = {}

    class FakeTensor:
        def float(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[3.0, 4.0, 0.0]], dtype=np.float32)

    class FakeModel:
        def encode_text(self, text, task, truncate_dim):
            calls["text"] = text
            calls["task"] = task
            calls["truncate_dim"] = truncate_dim
            return FakeTensor()

    cfg = Config(embedding_dim=3, device="cpu")
    model = FakeModel()

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    from src import search as search_mod

    monkeypatch.setattr(search_mod, "load_model", lambda _cfg: model)
    embedding = embed_text("a red car", cfg)

    assert embedding.shape == (3,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    np.testing.assert_allclose(embedding, [0.6, 0.8, 0.0])
    assert calls == {
        "text": "a red car",
        "task": "retrieval.query",
        "truncate_dim": 3,
    }


@pytest.mark.skip(reason="Implemented in Phase 2")
def test_build_and_search():
    """build_index() + search() should return the correct image in top-1."""
    pass