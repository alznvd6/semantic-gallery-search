"""Ranking metrics used by the terminal model-comparison runner."""
import math
import random
from statistics import mean
from typing import Dict, List, Sequence, Set, Tuple


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Return the fraction of relevant images retrieved in the top k."""
    if k <= 0 or not relevant_ids:
        return 0.0
    hits = set(ranked_ids[:k]) & relevant_ids
    return len(hits) / len(relevant_ids)


def average_precision(
    ranked_ids: Sequence[str], relevant_ids: Set[str], k: int
) -> float:
    """Return AP@k, normalized by min(total relevant images, k)."""
    if k <= 0 or not relevant_ids:
        return 0.0

    seen: Set[str] = set()
    hits = 0
    precision_sum = 0.0
    for rank, image_id in enumerate(ranked_ids[:k], start=1):
        if image_id in relevant_ids and image_id not in seen:
            seen.add(image_id)
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant_ids), k)


def ndcg_at_k(
    ranked_ids: Sequence[str], relevant_grades: Dict[str, int], k: int
) -> float:
    """Return normalized discounted cumulative gain at k."""
    if k <= 0:
        return 0.0

    def gain(grade: int, rank: int) -> float:
        return (2**grade - 1) / math.log2(rank + 1)

    actual = 0.0
    seen: Set[str] = set()
    for rank, image_id in enumerate(ranked_ids[:k], start=1):
        if image_id not in seen:
            actual += gain(relevant_grades.get(image_id, 0), rank)
            seen.add(image_id)
    ideal_grades = sorted(
        (grade for grade in relevant_grades.values() if grade > 0),
        reverse=True,
    )[:k]
    ideal = sum(
        gain(grade, rank) for rank, grade in enumerate(ideal_grades, start=1)
    )
    return actual / ideal if ideal else 0.0


def mrr(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    for rank, image_id in enumerate(ranked_ids, start=1):
        if image_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean."""
    if not values:
        raise ValueError("Cannot bootstrap an empty set of query scores.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if resamples < 1:
        raise ValueError("resamples must be positive.")

    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choice(values) for _ in values) for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    lower_index = int(tail * (resamples - 1))
    upper_index = int((1 - tail) * (resamples - 1))
    return samples[lower_index], samples[upper_index]
