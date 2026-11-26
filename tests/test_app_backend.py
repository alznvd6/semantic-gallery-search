import time
from threading import Event

import numpy as np
import pytest
from fastapi import HTTPException

from app.backend import api, search_service
from src import search as search_mod
from src.config import Config


def test_search_photos_builds_index_once_when_missing(monkeypatch, tmp_path):
    cfg = Config(
        index_path=tmp_path / "index.faiss",
        meta_path=tmp_path / "meta.json",
    )
    build_calls = []
    search_calls = []

    def build_index(config):
        build_calls.append(config)
        config.index_path.touch()
        config.meta_path.touch()

    def search(query, config, k):
        search_calls.append((query, config, k))
        return [("D:/gallery/img_0001.jpg", 0.9)]

    monkeypatch.setattr(search_service, "APP_CONFIG", cfg)
    monkeypatch.setattr(search_service.search_mod, "build_index", build_index)
    monkeypatch.setattr(search_service.search_mod, "search", search)

    expected = [("D:/gallery/img_0001.jpg", 0.9)]
    assert search_service.search_photos("a red car", k=5) == expected
    assert search_service.search_photos("a blue car", k=3) == expected
    assert build_calls == [cfg]
    assert search_calls == [
        ("a red car", cfg, 5),
        ("a blue car", cfg, 3),
    ]


def test_gallery_returns_real_image_urls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api,
        "list_photos",
        lambda: [tmp_path / "family trip 01.jpg", tmp_path / "garden.jpg"],
    )

    response = api.gallery()

    assert [photo.filename for photo in response.photos] == [
        "family trip 01.jpg",
        "garden.jpg",
    ]
    assert [photo.image_url for photo in response.photos] == [
        "/images/family%20trip%2001.jpg",
        "/images/garden.jpg",
    ]


def test_search_reports_model_runtime_errors_as_service_unavailable(monkeypatch):
    def fail_search(_query, k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(api, "search_photos", fail_search)

    with pytest.raises(HTTPException) as error:
        api.search(api.SearchRequest(query="a red car"))

    assert error.value.status_code == 503
    assert "CUDA out of memory" in error.value.detail


def test_build_index_endpoint_starts_a_background_build(monkeypatch, tmp_path):
    photos = [tmp_path / "img_0001.jpg", tmp_path / "img_0002.jpg"]
    started = []
    monkeypatch.setattr(api, "list_photos", lambda: photos)
    monkeypatch.setattr(
        api, "start_index_build", lambda: started.append(True) or True
    )

    response = api.build_index()

    assert response.status == "started"
    assert response.photo_count == 2
    assert started == [True]


def test_build_index_endpoint_rejects_an_empty_gallery(monkeypatch):
    monkeypatch.setattr(api, "list_photos", lambda: [])

    with pytest.raises(HTTPException) as error:
        api.build_index()

    assert error.value.status_code == 400


def test_start_index_build_skips_when_a_build_is_already_running(monkeypatch):
    flag = Event()
    flag.set()
    monkeypatch.setattr(search_service, "_index_build_in_progress", flag)

    assert search_service.start_index_build() is False


def test_search_ranks_matches_from_highest_to_lowest_score(monkeypatch):
    class FakeIndex:
        ntotal = 3

        def search(self, vector, k):
            return (
                np.array([[0.12, 0.88, 0.4]], dtype=np.float32),
                np.array([[0, 1, 2]]),
            )

    monkeypatch.setattr(
        search_mod,
        "load_index",
        lambda cfg: (
            FakeIndex(),
            {0: "a.jpg", 1: "b.jpg", 2: "c.jpg"},
        ),
    )
    monkeypatch.setattr(
        search_mod,
        "embed_text",
        lambda query, cfg: np.ones(4, dtype=np.float32),
    )

    results = search_mod.search("anything", Config(embedding_dim=4), k=3)

    assert [path for path, _ in results] == ["b.jpg", "c.jpg", "a.jpg"]
    assert [score for _, score in results] == pytest.approx([0.88, 0.4, 0.12])


def test_start_index_build_runs_the_build_in_the_background(monkeypatch):
    build_finished = Event()

    def build(config):
        build_finished.set()

    monkeypatch.setattr(search_service.search_mod, "build_index", build)

    assert search_service.start_index_build() is True
    assert build_finished.wait(5)
    deadline = time.time() + 2
    while search_service.index_building() and time.time() < deadline:
        time.sleep(0.01)
    assert not search_service.index_building()
