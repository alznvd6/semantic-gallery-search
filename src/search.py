"""
Core search module: embedding, indexing, retrieval.

"""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str) -> Any:
    """Load and cache a Jina CLIP model for the requested device."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def load_model(cfg: Config) -> Any:
    """Load the configured Jina CLIP model from Hugging Face.

    Args:
        cfg: project configuration.

    Returns:
        The evaluation-mode model, placed on cfg.device.
    """
    return _load_model(cfg.model_name, cfg.device)


def embed_text(text: str, cfg: Config) -> np.ndarray:
    """Convert a text query into an L2-normalized vector.

    Args:
        text: query string (Persian or English).
        cfg: project configuration.

    Returns:
        Vector of shape (cfg.embedding_dim,).
    """
    import torch

    model = load_model(cfg)
    with torch.inference_mode():
        embeddings = model.encode_text(
            text,
            task="retrieval.query",
            truncate_dim=cfg.embedding_dim,
        )

    vector = np.asarray(
        embeddings.float().detach().cpu().numpy(), dtype=np.float32
    ).reshape(-1)
    if vector.size != cfg.embedding_dim:
        raise ValueError(
            f"Expected a {cfg.embedding_dim}-dimensional text embedding, "
            f"got {vector.size} dimensions."
        )

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("The model returned a zero text embedding.")
    return vector / norm


def embed_images(image_paths: List[Path], cfg: Config) -> np.ndarray:
    """Convert a batch of images into L2-normalized vectors.

    Args:
        image_paths: list of image file paths.
        cfg: project configuration.

    Returns:
        Matrix of shape (len(image_paths), cfg.embedding_dim).

    TODO (Phase 2): load images, run image encoder in batches,
    L2-normalize each row.
    """
    raise NotImplementedError("embed_images: to be implemented in Phase 2")


def build_index(cfg: Config) -> None:
    """Build the FAISS index and metadata file from cfg.images_dir.

    Writes:
        cfg.index_path: FAISS IndexFlatIP with one vector per image.
        cfg.meta_path:  JSON mapping {row_id: image_path}.

    TODO (Phase 2): list images, embed them, add to FAISS,
    save index and metadata.
    """
    raise NotImplementedError("build_index: to be implemented in Phase 2")


def load_index(cfg: Config) -> Tuple[Any, Dict[int, str]]:
    """Load the FAISS index and metadata from disk.

    Args:
        cfg: project configuration.

    Returns:
        (faiss_index, {row_id: image_path}).

    TODO (Phase 2): read cfg.index_path and cfg.meta_path,
    return them as a tuple.
    """
    raise NotImplementedError("load_index: to be implemented in Phase 2")


def search(
    query: str,
    cfg: Config,
    k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Search the gallery for images matching a text query.

    Args:
        query: text description (Persian or English).
        cfg: project configuration.
        k: number of results; defaults to cfg.top_k.

    Returns:
        List of (image_path, similarity_score), sorted by score descending.

    TODO (Phase 2): embed query, search FAISS, map row ids
    back to image paths using metadata.
    """
    raise NotImplementedError("search: to be implemented in Phase 2")