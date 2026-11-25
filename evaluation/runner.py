"""Run a fixed-query comparison of configured image-text retrieval models."""
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from evaluation.metrics import (
    average_precision,
    bootstrap_mean_ci,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from src.config import Config
from src import search as search_mod


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
TOP_K = 10


def _read_models(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    models = config.get("models") if isinstance(config, dict) else None
    if not isinstance(models, list) or not 1 <= len(models) <= 6:
        raise ValueError("Configure between one and six models in models.json.")

    seen_names = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each model entry must be an object.")
        name = model.get("name")
        model_name = model.get("model_name")
        embedding_dim = model.get("embedding_dim", 1024)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each model entry needs a non-empty name.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"Model {name!r} needs a Hugging Face model_name.")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"Model {name!r} needs a positive embedding_dim.")
        model["name"] = name.strip()
        model["model_name"] = model_name.strip()
        model["embedding_dim"] = embedding_dim
        if model["model_name"] in seen_names:
            raise ValueError(f"Duplicate model_name: {model['model_name']}")
        seen_names.add(model["model_name"])
    return models


def _read_evaluation_data() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    cfg = Config()
    queries_path = PROJECT_ROOT / cfg.queries_path
    qrels_path = PROJECT_ROOT / cfg.qrels_path

    with queries_path.open(encoding="utf-8") as file:
        queries = json.load(file)
    with qrels_path.open(encoding="utf-8", newline="") as file:
        qrels_rows = csv.DictReader(file, delimiter="\t")
        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for row in qrels_rows:
            qrels[row["query_id"]][row["image_id"]] = int(row["relevance"])

    if not queries:
        raise ValueError("The fixed evaluation query set is empty.")
    missing_qrels = {
        query["query_id"] for query in queries if query["query_id"] not in qrels
    }
    if missing_qrels:
        raise ValueError(
            "Queries are missing relevance labels: "
            + ", ".join(sorted(missing_qrels))
        )
    return queries, dict(qrels)


def _model_config(model: Dict[str, Any], device: str) -> Config:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model["model_name"])
    artifact_dir = PROJECT_ROOT / "data" / "model-evaluation" / safe_name
    return Config(
        model_name=model["model_name"],
        embedding_dim=model["embedding_dim"],
        device=device,
        index_path=artifact_dir / "index.faiss",
        meta_path=artifact_dir / "meta.json",
    )


def _evaluate_model(
    model: Dict[str, Any],
    queries: List[Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]],
    device: str,
) -> Dict[str, Any]:
    cfg = _model_config(model, device)
    index_start = time.perf_counter()
    search_mod.build_index(cfg)
    index_build_seconds = time.perf_counter() - index_start

    query_results = []
    for query in queries:
        started = time.perf_counter()
        ranked_paths = search_mod.search(query["query"], cfg, k=TOP_K)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked_ids = [Path(path).name for path, _score in ranked_paths]
        grades = qrels[query["query_id"]]
        relevant_ids = {image_id for image_id, grade in grades.items() if grade > 0}
        query_results.append(
            {
                "query_id": query["query_id"],
                "language": query["language"],
                "type": query["type"],
                "recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
                "map@10": average_precision(ranked_ids, relevant_ids, TOP_K),
                "ndcg@10": ndcg_at_k(ranked_ids, grades, TOP_K),
                "mrr": mrr(ranked_ids, relevant_ids),
                "latency_ms": latency_ms,
            }
        )

    metrics = {
        name: mean(result[name] for result in query_results)
        for name in ("recall@5", "map@10", "ndcg@10", "mrr", "latency_ms")
    }
    metrics["recall@5_ci95"] = bootstrap_mean_ci(
        [result["recall@5"] for result in query_results]
    )
    metrics["mrr_ci95"] = bootstrap_mean_ci(
        [result["mrr"] for result in query_results]
    )
    metrics["recall@5_by_language"] = {
        language: mean(
            result["recall@5"]
            for result in query_results
            if result["language"] == language
        )
        for language in sorted({result["language"] for result in query_results})
    }
    metrics["recall@5_by_type"] = {
        query_type: mean(
            result["recall@5"]
            for result in query_results
            if result["type"] == query_type
        )
        for query_type in sorted({result["type"] for result in query_results})
    }
    return {
        "name": model["name"],
        "model_name": model["model_name"],
        "query_count": len(query_results),
        "index_build_seconds": index_build_seconds,
        "metrics": metrics,
        "per_query": query_results,
    }


def _print_results(results: List[Dict[str, Any]]) -> None:
    print()
    print(
        f"{'Model':32} {'R@5':>7} {'mAP@10':>8} {'nDCG@10':>9} "
        f"{'MRR':>7} {'ms/query':>10}"
    )
    print("-" * 82)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['name'][:32]:32} "
            f"{metrics['recall@5']:7.3f} "
            f"{metrics['map@10']:8.3f} "
            f"{metrics['ndcg@10']:9.3f} "
            f"{metrics['mrr']:7.3f} "
            f"{metrics['latency_ms']:10.1f}"
        )
        intervals = metrics["recall@5_ci95"]
        language_scores = ", ".join(
            f"{language} R@5={score:.3f}"
            for language, score in metrics["recall@5_by_language"].items()
        )
        print(
            f"  95% bootstrap CI for R@5: [{intervals[0]:.3f}, "
            f"{intervals[1]:.3f}], MRR CI: "
            f"[{metrics['mrr_ci95'][0]:.3f}, {metrics['mrr_ci95'][1]:.3f}]; "
            f"{language_scores}"
        )
        type_scores = ", ".join(
            f"{query_type} R@5={score:.3f}"
            for query_type, score in metrics["recall@5_by_type"].items()
        )
        print(f"  Query-type slices: {type_scores}")


def main() -> None:
    """Load fixed queries and compare configured models from the terminal."""
    parser = argparse.ArgumentParser(
        description="Compare image-text models on the fixed bilingual retrieval set."
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=EVALUATION_DIR / "models.json",
        help="JSON file containing one to six model entries.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=Config().device,
        help="Inference device; use the same device for every model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the complete results as JSON.",
    )
    args = parser.parse_args()

    try:
        models = _read_models(args.models)
        queries, qrels = _read_evaluation_data()
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    languages: Dict[str, int] = defaultdict(int)
    for query in queries:
        languages[query["language"]] += 1
    print(
        f"Fixed set: {len(queries)} queries, {len(qrels)} labeled queries; "
        + ", ".join(
            f"{language}={count}" for language, count in sorted(languages.items())
        )
    )
    if len(models) < 2:
        print("Only one model is configured; add candidates to compare models.")

    results = []
    for model in models:
        print(f"Evaluating {model['name']} ({model['model_name']})...")
        try:
            results.append(_evaluate_model(model, queries, qrels, args.device))
        except NotImplementedError as exc:
            parser.exit(
                2,
                f"Evaluation stopped: the search pipeline is not ready: {exc}\n",
            )

    _print_results(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
