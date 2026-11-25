"""Application-facing interface to the semantic search core."""
from typing import List, Optional, Tuple

from src import search as search_mod
from src.config import Config


def search_photos(
    query: str, k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """Search the configured photo gallery for a text query."""
    return search_mod.search(query, Config(), k=k)
