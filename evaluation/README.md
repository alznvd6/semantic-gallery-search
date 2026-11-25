# Terminal model evaluation

This folder is separate from the application UI. The comparison runner uses the
fixed queries and relevance labels in `data/eval/queries.json` and
`data/eval/qrels.tsv` for every candidate model.

## Configure models

Edit `models.json` and add up to six model entries. Keep the current Jina CLIP v2
entry as the baseline. Each entry has a display name, Hugging Face model ID, and
embedding dimension.

## Run

From the repository root:

```powershell
python evaluate.py --device cpu
```

Use `--device cuda` when CUDA is available. Keep the device the same across runs.
Optionally save the full per-query results:

```powershell
python evaluate.py --device cuda --output evaluation/results.json
```

The runner reports Recall@5, mAP@10, nDCG@10, MRR, average query latency, a
bootstrap confidence intervals for Recall@5 and MRR, and Recall@5 slices by
language and query type. The evaluation set currently contains 34 queries
(17 Persian, 17 English), with one relevance label per query. Treat conclusions
as preliminary and inspect the slices and confidence intervals before selecting
a model.

The current search/indexing functions in `src/search.py` are unfinished, and
model adapters may be required for candidates that do not expose the same
embedding interface. The runner reports this blocker instead of producing
placeholder scores.
