"""Central configuration for the semantic photo search system."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """All tunable parameters for the project."""

    model_name: str = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
    embedding_dim: int = 512
    device: str = "cuda"          # change to "cpu" if no GPU
    batch_size: int = 32

    images_dir: Path = Path("data/images")
    index_path: Path = Path("data/index.faiss")
    meta_path: Path = Path("data/meta.json")
    queries_path: Path = Path("data/eval/queries.json")
    qrels_path: Path = Path("data/eval/qrels.tsv")

    top_k: int = 10