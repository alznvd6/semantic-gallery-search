"""
Core search module: embedding, indexing, retrieval.

"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config


def load_model(cfg: Config) -> Tuple[Any, Any]:
    """Load the CLIP model and processor from HuggingFace.

    Args:
        cfg: project configuration.

    Returns:
        (model, processor) tuple.

    TODO (Phase 2): use transformers.CLIPModel / CLIPProcessor,
    move model to cfg.device, set to eval mode, cache the result.
    """
    raise NotImplementedError("load_model: to be implemented in Phase 2")


def embed_text(text: str, cfg: Config) -> np.ndarray:
    """Convert a text query into an L2-normalized vector.

    Args:
        text: query string (Persian or English).
        cfg: project configuration.

    Returns:
        Vector of shape (cfg.embedding_dim,).

    TODO (Phase 2): tokenize, run text encoder, L2-normalize.
    """
    raise NotImplementedError("embed_text: to be implemented in Phase 2")


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