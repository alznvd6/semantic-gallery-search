"""Central configuration for the semantic photo search system."""
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    """All tunable parameters for the project."""

    model_name: str = "jinaai/jina-clip-v2"
    embedding_dim: int = 1024
    device: str = "cuda"
    batch_size: int = 32

    images_dir: Path = PROJECT_ROOT / "data" / "images"
    index_path: Path = PROJECT_ROOT / "data" / "index.faiss"
    meta_path: Path = PROJECT_ROOT / "data" / "meta.json"
    queries_path: Path = PROJECT_ROOT / "data" / "eval" / "queries.json"
    qrels_path: Path = PROJECT_ROOT / "data" / "eval" / "qrels.tsv"

    top_k: int = 10