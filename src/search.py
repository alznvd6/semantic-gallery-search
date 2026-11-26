"""
Core search module: embedding, indexing, retrieval.

"""
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _model_family(model_name: str) -> str:
    if model_name == "Qwen/Qwen3-VL-Embedding-2B":
        return "qwen-vl"
    if model_name == "google/embeddinggemma-2":
        return "embeddinggemma"
    if "jina-clip-v2" in model_name.lower():
        return "jina"
    raise ValueError(f"No embedding adapter is registered for {model_name!r}.")


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str) -> Any:
    """Load and cache a supported model for the requested device."""
    family = _model_family(model_name)
    if family in ("qwen-vl", "embeddinggemma"):
        import torch
        from sentence_transformers import SentenceTransformer

        use_bfloat16 = (
            device.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": dtype},
        )
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=(family == "jina")
        )
        model.to(device)

    model.eval()
    return model


def load_model(cfg: Config) -> Any:
    """Load the configured embedding model in evaluation mode."""
    return _load_model(cfg.model_name, cfg.device)


def _as_numpy(embeddings: Any) -> np.ndarray:
    if hasattr(embeddings, "pooler_output"):
        embeddings = embeddings.pooler_output
        if embeddings is None:
            raise ValueError("The model returned no pooled embeddings.")
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach()
    if hasattr(embeddings, "float"):
        embeddings = embeddings.float()
    if hasattr(embeddings, "cpu"):
        embeddings = embeddings.cpu()
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()
    return np.asarray(embeddings, dtype=np.float32)


def _normalize_embeddings(
    embeddings: Any, expected_count: int, cfg: Config
) -> np.ndarray:
    matrix = _as_numpy(embeddings)
    if matrix.ndim == 1 and expected_count == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape != (expected_count, cfg.embedding_dim):
        raise ValueError(
            f"Expected embeddings with shape "
            f"({expected_count}, {cfg.embedding_dim}), got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("The model returned non-finite embeddings.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The model returned a zero embedding.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def embed_text(text: str, cfg: Config) -> np.ndarray:
    """Convert a text query into an L2-normalized vector."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The text query must not be empty.")

    import torch

    family = _model_family(cfg.model_name)
    model = load_model(cfg)
    with torch.inference_mode():
        if family == "jina":
            embeddings = model.encode_text(
                text,
                task="retrieval.query",
                truncate_dim=cfg.embedding_dim,
            )
        elif family == "qwen-vl":
            embeddings = model.encode(
                text.strip(),
                prompt="Retrieve relevant images for the user's text query.",
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        else:
            embeddings = model.encode(
                text.strip(),
                prompt_name="SearchQuery",
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

    return _normalize_embeddings(embeddings, 1, cfg)[0]


def embed_images(image_paths: List[Path], cfg: Config) -> np.ndarray:
    """Convert image files into an L2-normalized embedding matrix."""
    if not image_paths:
        return np.empty((0, cfg.embedding_dim), dtype=np.float32)
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    import torch

    family = _model_family(cfg.model_name)
    model = load_model(cfg)
    batches = []
    for offset in range(0, len(image_paths), cfg.batch_size):
        batch_paths = image_paths[offset : offset + cfg.batch_size]
        with torch.inference_mode():
            if family in ("qwen-vl", "embeddinggemma"):
                image_inputs = [
                    {"image": str(path.resolve())} for path in batch_paths
                ]
                embeddings = model.encode(
                    image_inputs,
                    batch_size=cfg.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            else:
                from PIL import Image

                images = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                if family == "jina":
                    embeddings = model.encode_image(
                        images, truncate_dim=cfg.embedding_dim
                    )

        batches.append(_normalize_embeddings(embeddings, len(batch_paths), cfg))

    return np.concatenate(batches, axis=0)


def _image_paths(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


@lru_cache(maxsize=1)
def _read_index(
    index_path: str,
    meta_path: str,
    index_modified_ns: int,
    metadata_modified_ns: int,
    embedding_dim: int,
) -> Tuple[Any, Dict[int, str]]:
    import faiss

    index = faiss.read_index(index_path)
    with Path(meta_path).open(encoding="utf-8") as file:
        raw_metadata = json.load(file)
    if not isinstance(raw_metadata, dict):
        raise ValueError("Search metadata must be a JSON object.")

    try:
        metadata = {int(row_id): path for row_id, path in raw_metadata.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Search metadata contains an invalid row ID.") from exc
    if any(not isinstance(path, str) for path in metadata.values()):
        raise ValueError("Search metadata image paths must be strings.")
    if set(metadata) != set(range(index.ntotal)):
        raise ValueError("Search metadata rows do not match the index.")
    if index.d != embedding_dim:
        raise ValueError(
            f"Index has dimension {index.d}; configured model expects "
            f"{embedding_dim}."
        )
    return index, metadata


def build_index(cfg: Config) -> None:
    """Build and persist a FAISS cosine-similarity index for the gallery."""
    import faiss

    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    image_paths = _image_paths(cfg.images_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in {cfg.images_dir}.")

    vectors = []
    for offset in range(0, len(image_paths), cfg.batch_size):
        vectors.append(
            embed_images(image_paths[offset : offset + cfg.batch_size], cfg)
        )
    embeddings = np.concatenate(vectors, axis=0)

    index = faiss.IndexFlatIP(cfg.embedding_dim)
    index.add(embeddings)
    metadata = {
        str(row_id): str(path.resolve())
        for row_id, path in enumerate(image_paths)
    }

    cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.meta_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cfg.index_path))
    with cfg.meta_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    _read_index.cache_clear()


def load_index(cfg: Config) -> Tuple[Any, Dict[int, str]]:
    """Load the FAISS index and its row-to-image-path metadata."""
    if not cfg.index_path.is_file():
        raise FileNotFoundError(
            f"Search index not found at {cfg.index_path}; build it first."
        )
    if not cfg.meta_path.is_file():
        raise FileNotFoundError(
            f"Search metadata not found at {cfg.meta_path}; build the index first."
        )

    index_stat = cfg.index_path.stat()
    metadata_stat = cfg.meta_path.stat()
    return _read_index(
        str(cfg.index_path),
        str(cfg.meta_path),
        index_stat.st_mtime_ns,
        metadata_stat.st_mtime_ns,
        cfg.embedding_dim,
    )


def search(
    query: str,
    cfg: Config,
    k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Return the top-k gallery images ranked by cosine similarity."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The text query must not be empty.")
    result_count = cfg.top_k if k is None else k
    if not isinstance(result_count, int) or result_count <= 0:
        raise ValueError("k must be a positive integer.")

    index, metadata = load_index(cfg)
    if index.ntotal == 0:
        return []

    vector = embed_text(query, cfg)
    scores, row_ids = index.search(
        np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32),
        min(result_count, index.ntotal),
    )
    return [
        (metadata[int(row_id)], float(score))
        for row_id, score in zip(row_ids[0], scores[0])
        if int(row_id) in metadata
    ]
