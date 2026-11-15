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
- **Streamlit UI** for interactive search
- **CLI** for batch indexing and search
- **Automated evaluation** with standard metrics (Recall@K, mAP, nDCG, MRR)
- **Modular architecture** — swap models by changing two lines in `config.py`

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Embedding Model | M-CLIP (XLM-RoBERTa-Large + ViT-B/32) | Converts text and images into a shared vector space |
| Vector Index | FAISS (`IndexFlatIP`) | Cosine similarity search |
| Framework | PyTorch + Hugging Face Transformers | Model loading and inference |
| UI | Streamlit | Interactive web app |
| Testing | pytest | Unit tests |

---

## Project Structure

_Content not included in the original excerpt._