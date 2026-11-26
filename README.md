# Semantic Photo Search

Semantic text-to-image search over a personal image gallery (~300 photos).  
Describe what you remember — find the closest matching photos.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Selection](#model-selection)
- [Evaluation](#evaluation)
- [Development Status](#development-status)
- [Resources](#resources)

---

## Features

- **Text-to-image search** in Persian and English
- **Vector indexing** with FAISS for fast search (millisecond-level)
- **React UI** for interactive search
- **CLI** for batch indexing and search
- **Automated evaluation** with standard metrics (Recall@K, mAP, nDCG, MRR)
- **Modular architecture** — swap models by changing two lines in `config.py`

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Embedding Model | Jina CLIP v2 | Converts multilingual text and images into a shared 1,024-dimensional vector space |
| Vector Index | FAISS (`IndexFlatIP`) | Cosine similarity search |
| Framework | PyTorch + Hugging Face Transformers | Model loading and inference |
| UI | React, HTML, and CSS | Web frontend |
| Testing | pytest | Unit tests |

---

## Project Structure

```text
app/
  backend/
    api.py              # FastAPI HTTP API
    search_service.py   # Application-facing search service
  frontend/
    index.html
    package.json        # Frontend dependencies and scripts
    vite.config.js
    ui-designs/         # Standalone HTML design prototypes
      index.html
      terracotta.html
    src/
      App.jsx
      main.jsx
      styles.css
evaluation/             # Terminal-only model comparison
  models.json
  runner.py
  metrics.py
src/
  config.py             # Search configuration
  search.py             # Embedding, indexing, and retrieval core
cli.py                  # Command-line interface
evaluate.py             # Terminal model-evaluation entry point
data/
  images/               # Photo gallery
tests/
```

## Running the Application

Install the frontend packages:

```powershell
cd app/frontend
npm install
```

Start the backend from the repository root. The launcher checks CUDA and the
cached Jina weights, then runs the backend in the existing `dl_env` Conda
environment:

```powershell
.\run_app_backend.ps1
```

In another terminal, start the React frontend:

```powershell
cd app/frontend
npm run dev
```

The frontend is in `app/frontend/`; its FastAPI backend is in `app/backend/`.
The backend uses Jina CLIP v2 on CUDA to index and search the local image
gallery. Open `http://127.0.0.1:5173` to use the Persian RTL interface. The
gallery appears from the local image folder immediately; the first search
loads Jina and builds the FAISS index, which can take several minutes. Later
searches reuse the saved index at `data/index.faiss`.

To compare standalone HTML designs with styles and color palettes kept inside
each file, open `http://localhost:5173/ui-designs/` while the frontend dev server
is running. Add or duplicate HTML files in `app/frontend/ui-designs/` to try
more variations.

Model comparison is separate from the application and runs in the terminal.
Configure candidates in `evaluation/models.json`, then run
`python evaluate.py --device cpu` from the repository root. See
`evaluation/README.md` for metrics, outputs, and current limitations.
