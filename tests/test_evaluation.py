"""Focused tests for terminal retrieval evaluation metrics."""
import json
import sys

import pytest

from evaluation.metrics import (
    average_precision,
    bootstrap_mean_ci,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from evaluation.runner import _model_config, _read_evaluation_data, _select_models


def test_recall_at_k_is_fraction_of_relevant_results():
    assert recall_at_k(["a.jpg", "b.jpg"], {"b.jpg", "c.jpg"}, 2) == 0.5
    assert recall_at_k(["a.jpg"], set(), 5) == 0.0


def test_average_precision_at_k_uses_relevant_rank_positions():
    assert average_precision(
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        {"b.jpg", "d.jpg"},
        4,
    ) == pytest.approx(0.5)
    assert average_precision(["a.jpg"], set(), 10) == 0.0


def test_ndcg_is_one_for_ideal_ranking_and_lower_when_reversed():
    grades = {"a.jpg": 2, "b.jpg": 1}
    assert ndcg_at_k(["a.jpg", "b.jpg"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b.jpg", "a.jpg"], grades, 2) < 1.0
    assert ndcg_at_k(["a.jpg", "a.jpg"], grades, 2) < 1.0


def test_mrr_returns_first_relevant_reciprocal_rank():
    assert mrr(["a.jpg", "b.jpg"], {"b.jpg"}) == 0.5
    assert mrr(["a.jpg"], {"b.jpg"}) == 0.0


def test_bootstrap_mean_interval_is_repeatable():
    first = bootstrap_mean_ci([0.0, 0.5, 1.0], resamples=200, seed=7)
    second = bootstrap_mean_ci([0.0, 0.5, 1.0], resamples=200, seed=7)
    assert first == second
    assert first[0] <= 0.5 <= first[1]


def test_fixed_queries_have_relevance_labels_and_both_languages():
    queries, qrels = _read_evaluation_data()
    query_ids = {query["query_id"] for query in queries}
    assert query_ids == set(qrels)
    assert {query["language"] for query in queries} == {"en", "fa"}


def test_model_filter_selects_only_the_requested_configured_model():
    models = [
        {"model_name": "model/first"},
        {"model_name": "model/second"},
    ]

    assert _select_models(models, "model/second") == [models[1]]
    with pytest.raises(ValueError, match="is not configured"):
        _select_models(models, "model/missing")


def test_model_config_uses_requested_batch_size(tmp_path, monkeypatch):
    from evaluation import runner

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    cfg = _model_config(
        {"model_name": "model/first", "embedding_dim": 3},
        device="cpu",
        batch_size=1,
    )

    assert cfg.batch_size == 1


def test_model_evaluation_result_keeps_source_query_and_ranked_judgements(
    monkeypatch, tmp_path
):
    from evaluation import runner
    from src.config import Config

    monkeypatch.setattr(
        runner,
        "_model_config",
        lambda _model, _device, _batch_size: Config(
            model_name="model/first",
            embedding_dim=2,
            device="cpu",
            index_path=tmp_path / "index.faiss",
            meta_path=tmp_path / "meta.json",
        ),
    )
    monkeypatch.setattr(runner.search_mod, "build_index", lambda _cfg: None)
    monkeypatch.setattr(
        runner.search_mod,
        "search",
        lambda _query, _cfg, k: [
            ("C:/gallery/img_0001.jpg", 0.9),
            ("C:/gallery/img_0002.jpg", 0.6),
        ][:k],
    )
    query = {
        "query_id": "q001",
        "query": "a Persian test query",
        "language": "fa",
        "type": "simple",
    }

    result = runner._evaluate_model(
        {"name": "First", "model_name": "model/first", "embedding_dim": 2},
        [query],
        {"q001": {"img_0001.jpg": 2}},
        device="cpu",
        batch_size=1,
    )

    per_query = result["per_query"][0]
    assert per_query["query"] == query["query"]
    assert per_query["relevant_images"] == [
        {"image_id": "img_0001.jpg", "relevance": 2}
    ]
    assert per_query["top_results"] == [
        {
            "rank": 1,
            "image_id": "img_0001.jpg",
            "score": 0.9,
            "relevance": 2,
        },
        {
            "rank": 2,
            "image_id": "img_0002.jpg",
            "score": 0.6,
            "relevance": 0,
        },
    ]


def test_main_saves_query_sources_settings_and_model_results(monkeypatch, tmp_path):
    from evaluation import runner

    model = {
        "name": "First",
        "model_name": "model/first",
        "embedding_dim": 2,
    }
    query = {
        "query_id": "q001",
        "query": "the exact test query",
        "language": "en",
        "type": "simple",
    }
    model_result = {"name": "First", "per_query": [{"query": query["query"]}]}
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--device",
        "cpu",
        "--model-name",
        model["model_name"],
        "--batch-size",
        "1",
        "--output",
        str(output),
    ])
    monkeypatch.setattr(runner, "_read_models", lambda _path: [model])
    monkeypatch.setattr(
        runner,
        "_read_evaluation_data",
        lambda: ([query], {"q001": {"img_0001.jpg": 2}}),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_model",
        lambda *_args: model_result,
    )
    monkeypatch.setattr(runner, "_print_results", lambda _results: None)

    runner.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation"]["queries_file"] == "data/eval/queries.json"
    assert report["evaluation"]["qrels_file"] == "data/eval/qrels.tsv"
    assert report["evaluation"]["query_count"] == 1
    assert report["evaluation"]["device"] == "cpu"
    assert report["evaluation"]["batch_size"] == 1
    assert report["models"] == [model_result]
