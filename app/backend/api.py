"""HTTP API for the semantic photo search application."""
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.backend.search_service import (
    APP_CONFIG,
    index_building,
    index_ready,
    index_stale,
    indexed_count,
    list_photos,
    search_photos,
    start_index_build,
)


app = FastAPI(title="Semantic Photo Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
APP_CONFIG.images_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/images",
    StaticFiles(directory=str(APP_CONFIG.images_dir)),
    name="images",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=30)


class SearchResult(BaseModel):
    filename: str
    score: float
    image_url: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


class GalleryPhoto(BaseModel):
    filename: str
    image_url: str
    date: Optional[str] = None


class GalleryResponse(BaseModel):
    photos: List[GalleryPhoto]


class IndexResponse(BaseModel):
    status: str
    photo_count: int


def _photo_date(path: Path) -> Optional[str]:
    """Return the photo's last-modified date, when the file is readable."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
    except OSError:
        return None


@app.get("/api/health")
def health() -> dict:
    """Report the selected model, gallery size, and index state."""
    photos = list_photos()
    return {
        "status": "ok",
        "model": APP_CONFIG.model_name,
        "device": APP_CONFIG.device,
        "index_ready": index_ready(),
        "indexing": index_building(),
        "index_stale": index_stale(),
        "photo_count": len(photos),
        "indexed_count": indexed_count(),
    }


@app.get("/api/gallery", response_model=GalleryResponse)
def gallery() -> GalleryResponse:
    """Return image files available in the user's local gallery."""
    return GalleryResponse(
        photos=[
            GalleryPhoto(
                filename=path.name,
                image_url=f"/images/{quote(path.name)}",
                date=_photo_date(path),
            )
            for path in list_photos()
        ]
    )


@app.post("/api/index", response_model=IndexResponse)
def build_index() -> IndexResponse:
    """Embed every gallery image and cache the vectors for later searches."""
    try:
        photos = list_photos()
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"خواندن پوشه‌ی گالری ممکن نشد: {exc}",
        ) from exc

    if not photos:
        raise HTTPException(
            status_code=400,
            detail="هیچ عکسی در پوشه‌ی گالری پیدا نشد.",
        )

    started = start_index_build()
    return IndexResponse(
        status="started" if started else "running",
        photo_count=len(photos),
    )


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Search the gallery and return image paths and similarity scores."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    try:
        matches = search_photos(query, k=request.limit)
    except (
        FileNotFoundError,
        ImportError,
        NotImplementedError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"جستجوی معنایی در دسترس نیست: {exc}",
        ) from exc

    results = []
    for path, score in matches:
        filename = Path(path).name
        results.append(
            SearchResult(
                filename=filename,
                score=score,
                image_url=f"/images/{quote(filename)}",
            )
        )
    return SearchResponse(results=results)
