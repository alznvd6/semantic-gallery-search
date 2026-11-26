"""Central configuration for the semantic photo search system."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """All tunable parameters for the project."""

    model_name: str = "jinaai/jina-clip-v2"
    embedding_dim: int = 1024
    device: str = "cuda"          
    batch_size: int = 32

    images_dir: Path = Path("D:\\semantic-photo-search\\data\\images")
    index_path: Path = Path("D:\\semantic-photo-search\\data\\index.faiss")
    meta_path: Path = Path("D:\\semantic-photo-search\\data\\meta.json")
    queries_path: Path = Path("D:\\semantic-photo-search\\data\\eval\\queries.json")
    qrels_path: Path = Path("D:\\semantic-photo-search\\data\\eval\\qrels.tsv")

    top_k: int = 10