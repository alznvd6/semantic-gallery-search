"""Backward-compatible terminal entry point for model evaluation."""
from evaluation.metrics import (
    average_precision,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from evaluation.runner import main


if __name__ == "__main__":
    main()