# LLM Agentic Legal Information Retrieval

Kaggle competition entry for cross-lingual Swiss legal citation retrieval. Given English legal questions, the system retrieves relevant Swiss legal citations (statutes and court decisions, mostly in German) and outputs `submission.csv`. Scored by Macro-F1.

## Pipeline Overview

**12-stage retrieval pipeline:**

| Stage | Method | Weight | Description |
|-------|--------|--------|-------------|
| 1 | Regex extraction | 10.0 | Direct citation parsing (Art., BGE, case numbers) |
| 2 | Abbreviation expansion | 3.0 | Expand matched law abbreviations (e.g. StGB, OR) |
| 3 | Domain keywords | 1.5 | Map legal topics to likely law codes |
| 4 | Query TF-IDF transfer | 1.5 | Find similar training queries, borrow their citations |
| 5 | Global frequency boost | 0.03 | Micro-boost for frequently cited laws |
| 6 | Co-citation (train) | 0.005 | Expand via citation co-occurrence in training data |
| 7 | Co-citation (court) | 2.0 | Expand via citation co-occurrence in 2.47M court decisions |
| 8 | Bridge expansion | 1.5 | Link laws to court decisions that cite them |
| 9 | Laws text TF-IDF | 1.0 | Match translated query against law text content |
| 10 | BGE-M3 dense retrieval | 2.5 | Multilingual dense embeddings on 175K laws via FAISS |
| 11 | BM25 laws (stemmed) | 2.0 | German-stemmed BM25 on laws corpus |
| 12 | BM25 court (stemmed) | 1.5 | German-stemmed BM25 on court corpus |

Final candidates are reranked by a cross-encoder (BAAI/bge-reranker-v2-m3) with 0.7/0.3 interpolation.

## Project Structure

```
.
├── notebook_kaggle.ipynb        # Main pipeline notebook (Kaggle / Vast.ai)
├── build_notebook.py            # Programmatic notebook builder from cells
├── vastai_setup.sh              # Vast.ai instance setup script
├── README.md
├── CLAUDE.md                    # AI assistant project instructions
│
├── competitor_notebooks/        # Scraped competitor solutions
│   ├── citation_graph/          # Citation graph approach
│   ├── hybrid_baseline/         # Hybrid retrieval baseline
│   ├── qwen3_0065/              # Qwen3 embedding + reranker (LB 0.065)
│   ├── tfidf_cocitation/        # TF-IDF + co-citation approach
│   └── tfidf-cocitation.ipynb   # Top competitor (LB 0.0603)
│
├── scripts/                     # Notebook cell builders & patching utilities
├── _cells/                      # Individual cell source files
├── config/                      # Kaggle kernel metadata
│
├── Data/                        # Competition data (not in repo)
│   ├── court_considerations.csv # 2.47M court decisions (2.4GB)
│   ├── laws_de.csv              # 175K Swiss law articles (73MB)
│   ├── train.csv                # Training queries with gold citations
│   ├── val.csv                  # Validation queries
│   └── test.csv                 # Test queries (predict these)
│
├── models/                      # HuggingFace models (not in repo)
├── kaggle_dataset/              # Packaged dataset for Kaggle upload
├── kaggle_submission/           # Generated submission files
│
├── archive/                     # Old iterations & experiments
│   ├── finetune_proper*.ipynb   # Fine-tuning experiments (dead end)
│   ├── llm-agentic-legal-*.ipynb # Earlier pipeline versions
│   └── ...                      # Logs, reports, broken versions
│
└── .planning/                   # Project planning artifacts
    ├── ROADMAP.md               # Phase breakdown
    ├── IMPLEMENTATION_PLAN.md   # Detailed implementation plan
    └── phases/                  # Per-phase research & execution
```

## Models Used

- **Qwen/Qwen2.5-7B-Instruct** - EN-to-DE translation (fp16)
- **BAAI/bge-m3** - Multilingual dense retrieval encoder (560MB)
- **BAAI/bge-reranker-v2-m3** - Cross-encoder reranker (120MB)

## Runtime Environments

| Environment | GPU | Data Source | Notes |
|------------|-----|-------------|-------|
| **Kaggle** | T4 x2 (16GB each) | Pre-uploaded dataset | Offline, 12-hour limit |
| **Vast.ai** | RTX 5090 (32GB) | Kaggle API download | Internet access, pay-per-hour |
| **Local** | CPU/GPU | `Data/` directory | Development and debugging |

## Quick Start

1. Upload models to Kaggle as datasets (or let Vast.ai download from HuggingFace)
2. Run all cells in `notebook_kaggle.ipynb`
3. Output: `submission.csv` with predicted citations per query
