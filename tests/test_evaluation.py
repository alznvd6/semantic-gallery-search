"""Focused tests for terminal retrieval evaluation metrics."""
import pytest

from evaluation.metrics import (
    average_precision,
    bootstrap_mean_ci,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from evaluation.runner import _read_evaluation_data


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
