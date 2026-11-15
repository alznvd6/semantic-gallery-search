"""Evaluation metrics and runner.

Phase 0: only signatures.
Phase 4: full implementation.

Metrics to implement:
    - recall_at_k
    - average_precision (mAP)
    - ndcg_at_k
    - mrr
"""
import argparse
from typing import Dict, List


def recall_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    """Return 1.0 if any relevant id is in the top-k, else 0.0.

    TODO (Phase 4): currently implemented as a simple check.
    """
    return 1.0 if any(i in relevant_ids for i in ranked_ids[:k]) else 0.0


def average_precision(
    ranked_ids: List[str], relevant_ids: set, k: int
) -> float:
    """Compute average precision at k for a single query.

    TODO (Phase 4).
    """
    raise NotImplementedError("average_precision: to be implemented in Phase 4")


def ndcg_at_k(
    ranked_ids: List[str], relevant_grades: Dict[str, int], k: int
) -> float:
    """Compute nDCG@k for a single query.

    TODO (Phase 4).
    """
    raise NotImplementedError("ndcg_at_k: to be implemented in Phase 4")


def mrr(ranked_ids: List[str], relevant_ids: set) -> float:
    """Return the reciprocal rank of the first relevant result.

    TODO (Phase 4).
    """
    raise NotImplementedError("mrr: to be implemented in Phase 4")


def main() -> None:
    """Parse args and run evaluation over queries.json + qrels.tsv.

    TODO (Phase 4): read queries, run search, compute all metrics,
    write evaluation_report.md.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="print successful and failed examples",
    )
    parser.parse_args()
    print("Evaluation runner: to be implemented in Phase 4.")


if __name__ == "__main__":
    main()