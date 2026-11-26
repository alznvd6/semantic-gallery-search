"""Application-facing interface to the semantic search core."""
import json
from threading import Event, Lock, Thread
from pathlib import Path
from typing import List, Optional, Tuple

from src import search as search_mod
from src.config import Config


APP_CONFIG = Config(batch_size=1)
_index_build_lock = Lock()
_index_start_lock = Lock()
_index_build_in_progress = Event()


def list_photos() -> List[Path]:
    """List the real images available in the configured gallery."""
    return search_mod.list_images(APP_CONFIG.images_dir)


def index_ready() -> bool:
    """Return whether both persisted index files are available."""
    return APP_CONFIG.index_path.is_file() and APP_CONFIG.meta_path.is_file()


def index_building() -> bool:
    """Return whether the index is being (re)built right now."""
    return _index_build_in_progress.is_set()


def _indexed_paths() -> List[str]:
    """Read the image paths recorded in the index metadata."""
    try:
        with APP_CONFIG.meta_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, ValueError):
        return []
    if not isinstance(metadata, dict):
        return []
    return [path for path in metadata.values() if isinstance(path, str)]


def indexed_count() -> int:
    """Return how many images the persisted index covers."""
    return len(_indexed_paths())


def index_stale() -> bool:
    """Return whether the gallery changed after the index was written."""
    if not index_ready():
        return False
    try:
        index_modified_ns = APP_CONFIG.index_path.stat().st_mtime_ns
        image_modified = [
            path.stat().st_mtime_ns for path in list_photos()
        ]
    except OSError:
        return False
    if any(modified > index_modified_ns for modified in image_modified):
        return True
    return any(not Path(path).is_file() for path in _indexed_paths())


def _build_index_locked(force: bool) -> None:
    """Build and persist the index while holding the build lock."""
    with _index_build_lock:
        if not force and index_ready() and not index_stale():
            return
        _index_build_in_progress.set()
        try:
            search_mod.build_index(APP_CONFIG)
        finally:
            _index_build_in_progress.clear()


def _ensure_index() -> None:
    """Build the index only when it is missing or out of date."""
    _build_index_locked(force=False)


def rebuild_index() -> None:
    """Embed every gallery image and cache the vectors, blocking until done."""
    _build_index_locked(force=True)


def start_index_build() -> bool:
    """Start a full index build in the background; False if one is running."""
    with _index_start_lock:
        if _index_build_in_progress.is_set():
            return False
        _index_build_in_progress.set()

    def run() -> None:
        try:
            with _index_build_lock:
                search_mod.build_index(APP_CONFIG)
        finally:
            _index_build_in_progress.clear()

    Thread(target=run, name="gallery-index", daemon=True).start()
    return True


def search_photos(
    query: str, k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """Search the configured Jina gallery, building its index on first use."""
    _ensure_index()
    return search_mod.search(query, APP_CONFIG, k=k)
