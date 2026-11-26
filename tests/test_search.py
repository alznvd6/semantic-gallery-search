"""Focused tests for model adapters and semantic search indexing."""
import sys
from contextlib import nullcontext
from pathlib import Path
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


@pytest.mark.parametrize(
    ("device", "bf16_supported", "expected_dtype"),
    [
        ("cpu", True, "float32"),
        ("cuda", False, "float32"),
        ("cuda", True, "bfloat16"),
    ],
)
def test_sentence_transformer_models_use_safe_precision(
    monkeypatch, request, device, bf16_supported, expected_dtype
):
    _load_model.cache_clear()
    request.addfinalizer(_load_model.cache_clear)
    calls = {}

    class FakeModel:
        def eval(self):
            calls["eval"] = True

    model = FakeModel()

    class FakeSentenceTransformer:
        def __new__(self, model_name, device, model_kwargs):
            calls.update(
                model_name=model_name,
                device=device,
                dtype=model_kwargs["torch_dtype"],
            )
            return model

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: device == "cuda",
            is_bf16_supported=lambda: bf16_supported,
        ),
        float32="float32",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    loaded = load_model(
        Config(model_name="google/embeddinggemma-2", device=device)
    )

    assert loaded is model
    assert calls == {
        "model_name": "google/embeddinggemma-2",
        "device": device,
        "dtype": expected_dtype,
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


@pytest.mark.parametrize(
    ("model_name", "expected_argument"),
    [
        (
            "Qwen/Qwen3-VL-Embedding-2B",
            ("prompt", "Retrieve relevant images for the user's text query."),
        ),
        ("google/embeddinggemma-2", ("prompt_name", "SearchQuery")),
    ],
)
def test_sentence_transformer_candidate_query_prompts(
    monkeypatch, model_name, expected_argument
):
    class FakeModel:
        def encode(self, text, **kwargs):
            assert text == "a red car"
            assert kwargs[expected_argument[0]] == expected_argument[1]
            return np.array([3.0, 4.0], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    from src import search as search_mod

    monkeypatch.setattr(search_mod, "load_model", lambda _cfg: FakeModel())
    cfg = Config(model_name=model_name, embedding_dim=2, device="cpu")
    vector = embed_text("a red car", cfg)

    np.testing.assert_allclose(vector, [0.6, 0.8])


@pytest.mark.parametrize(
    "model_name",
    [
        "jinaai/jina-clip-v2",
        "Qwen/Qwen3-VL-Embedding-2B",
        "google/embeddinggemma-2",
    ],
)
def test_image_adapters_return_normalized_vectors(monkeypatch, tmp_path, model_name):
    from PIL import Image

    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (2, 2), color="red").save(image_path)
    calls = {}

    class FakeTensor:
        def to(self, _device):
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[3.0, 4.0]], dtype=np.float32)

    class FakeModel:
        def encode_image(self, images, truncate_dim):
            calls["jina"] = (len(images), truncate_dim)
            return FakeTensor()

        def encode(self, inputs, **kwargs):
            assert len(inputs) == 1
            assert inputs[0]["image"] == str(image_path.resolve())
            calls["sentence_transformer"] = True
            return np.array([[3.0, 4.0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    from src import search as search_mod

    monkeypatch.setattr(search_mod, "load_model", lambda _cfg: FakeModel())
    cfg = Config(
        model_name=model_name,
        embedding_dim=2,
        batch_size=1,
        device="cpu",
    )

    embeddings = search_mod.embed_images([image_path], cfg)

    assert embeddings.shape == (1, 2)
    np.testing.assert_allclose(embeddings[0], [0.6, 0.8])
    if model_name == "jinaai/jina-clip-v2":
        assert calls == {"jina": (1, 2)}
    else:
        assert calls == {"sentence_transformer": True}


def test_build_and_search_returns_highest_similarity_image(monkeypatch, tmp_path):
    from src import search as search_mod

    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"image a")
    image_b.write_bytes(b"image b")
    cfg = Config(
        images_dir=tmp_path,
        index_path=tmp_path / "index.faiss",
        meta_path=tmp_path / "meta.json",
        embedding_dim=2,
        device="cpu",
    )
    stored_indexes = {}
    index_reads = []

    class FakeIndex:
        def __init__(self, dimension):
            self.d = dimension
            self.vectors = np.empty((0, dimension), dtype=np.float32)

        @property
        def ntotal(self):
            return len(self.vectors)

        def add(self, vectors):
            self.vectors = np.concatenate((self.vectors, vectors), axis=0)

        def search(self, queries, count):
            scores = queries @ self.vectors.T
            rows = np.argsort(-scores[0])[:count]
            return scores[:, rows], rows.reshape(1, -1)

    def write_index(index, path):
        stored_indexes[path] = index
        Path(path).write_bytes(b"index")

    def read_index(path):
        index_reads.append(path)
        return stored_indexes[path]

    fake_faiss = SimpleNamespace(
        IndexFlatIP=FakeIndex,
        write_index=write_index,
        read_index=read_index,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setattr(
        search_mod,
        "embed_images",
        lambda paths, _cfg: np.array(
            [
                [1.0, 0.0] if path.name == "a.jpg" else [0.0, 1.0]
                for path in paths
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        search_mod,
        "embed_text",
        lambda _query, _cfg: np.array([0.0, 1.0], dtype=np.float32),
    )

    search_mod.build_index(cfg)
    results = search_mod.search("a red car", cfg, k=1)
    search_mod.search("a blue car", cfg, k=1)

    assert results == [(str(image_b.resolve()), pytest.approx(1.0))]
    assert len(index_reads) == 1