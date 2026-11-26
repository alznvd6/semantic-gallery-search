# Terminal model evaluation

This folder is separate from the application UI. The comparison runner uses the
fixed queries and relevance labels in `data/eval/queries.json` and
`data/eval/qrels.tsv` for every candidate model.

## Configure models

Edit `models.json` and add up to six model entries. Keep the current Jina CLIP v2
entry as the baseline. Qwen3-VL-Embedding-2B and EmbeddingGemma 2 are
preconfigured candidates. Each entry has a display name,
Hugging Face model ID, and embedding dimension.

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

Every run reads the exact query text from `data/eval/queries.json` and relevance
labels from `data/eval/qrels.tsv`. The saved JSON report records both source
paths and run settings, and includes the original text, known relevant images,
retrieved image IDs, similarity scores, and per-query metrics for every query.

The evaluation runner rebuilds a separate image index for each model before
scoring queries. Qwen3-VL-Embedding-2B and EmbeddingGemma 2 use the
`sentence-transformers` multimodal interface; Jina CLIP v2 uses its Transformers
model interface. Model weights are downloaded from Hugging Face
the first time each model is loaded, so allow for substantial download time and
memory use.

The configured checkpoints are cached on the E: drive. In PowerShell, point
Hugging Face at that cache before running evaluation:

```powershell
$env:HF_HOME = "E:\semantic-photo-search-models"
python evaluate.py --device cuda --output evaluation/results.json
```

For CPU-only runs, use the per-model scripts. They check free RAM before
starting, use batch size 1, keep the Python environment and result files on E:,
and install the CPU requirements there on first use:

```powershell
.\evaluation\run_jina_cpu.ps1
.\evaluation\run_embeddinggemma_cpu.ps1
# Run separately after the other candidates:
.\evaluation\run_qwen_cpu.ps1
```

The RAM checks are conservative estimates, not guarantees. Each run writes a
timestamped result JSON under `E:\semantic-photo-search-results`.

For Jina CLIP v2 on an NVIDIA GPU, run:

```powershell
.\evaluation\run_jina_gpu.ps1
```

The script activates the existing Conda environment `dl_env` and verifies that
CUDA-enabled PyTorch and Transformers `>=4.57,<5` are available; it does not
install or modify packages. The Transformers range is required by Jina's cached
model code. It requires an NVIDIA driver, at least 6 GB free system RAM, and at
least 3 GB free GPU memory; it runs with batch size 1 and writes a timestamped
result JSON under `E:\semantic-photo-search-results`.
