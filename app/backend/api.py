"""HTTP API for the semantic photo search application."""
from pathlib import Path
from typing import List
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.backend.search_service import search_photos
from src.config import Config


project_root = Path(__file__).resolve().parents[2]
images_dir = Config().images_dir
if not images_dir.is_absolute():
    images_dir = project_root / images_dir

app = FastAPI(title="Semantic Photo Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=30)


class SearchResult(BaseModel):
    filename: str
    score: float
    image_url: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.get("/api/health")
def health() -> dict:
    """Report that the API process is ready."""
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Search the gallery and return image paths and similarity scores."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    try:
        matches = search_photos(query, k=request.limit)
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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
