# Codebase Structure

**Analysis Date:** 2026-04-10

## Directory Layout

```
LLm_Agentic/
├── .planning/                           # GSD planning documents
│   └── codebase/
│       ├── STACK.md
│       └── ARCHITECTURE.md
├── .git/                                # Git history (not committed for size)
├── Data/                                # Input data (laws, court, val, test CSVs)
├── models/                              # Pretrained models downloaded locally
│   ├── multilingual-e5-large/           # Dense encoder (~560MB)
│   ├── multilingual-e5-small/           # Alternative smaller encoder
│   ├── mmarco-reranker/                 # Cross-encoder reranker (~120MB)
│   └── opus-mt-en-de/                   # English-to-German translator (~300MB)
├── kernel_output/                       # Output from Kaggle kernel runs
│
├── solution.py                          # MAIN: Full hybrid retrieval pipeline
├── notebook_kaggle.ipynb                # MAIN: Kaggle notebook version (cells)
├── eval_local.py                        # TF-IDF baseline evaluation
├── eval_dense.py                        # Dense retrieval evaluation
├── download_models.py                   # Utility: Download models from HF Hub
│
├── diagnose.py                          # Verify gold citations in corpus
├── diagnose2.py                         # Diagnostic variant 2
├── diagnose3.py                         # Diagnostic variant 3
├── diagnose4.py                         # Diagnostic variant 4
├── diagnose5.py                         # Diagnostic variant 5
├── diagnose6.py                         # kNN on train queries baseline
│
├── watch_and_submit.sh                  # Shell script: Auto-run and submit to Kaggle
├── watch_simple.sh                      # Shell script: Simpler watcher
│
├── requirements.txt                     # Python dependencies
├── SETUP.md                             # Setup instructions
├── Overview.md                          # Competition overview (from Kaggle)
├── AboutData.md                         # Data format documentation
├── Competition_Analysis_Report.docx     # Analysis notes
├── kernel-metadata.json                 # Kaggle notebook metadata
└── kaggle.json                          # Kaggle API credentials
```

## Directory Purposes

**Data/:**
- Purpose: Contains input CSV files for the retrieval corpus and query sets
- Contains: laws_de.csv, court_considerations.csv, train.csv, val.csv, test.csv
- Key files: 
  - `laws_de.csv` - ~175K Swiss law documents with citation + text
  - `court_considerations.csv` - ~2.47M court decision excerpts with citation + text
  - `val.csv` - ~10 validation queries with gold citations (for calibration)
  - `test.csv` - ~10 test queries (no gold; submission target)
  - `train.csv` - ~1140 training queries with gold citations

**models/:**
- Purpose: Local cache of pretrained transformer models (for offline development)
- Contains: Model checkpoints and weights from Hugging Face Hub
- Key files: 
  - `multilingual-e5-large/` - Encoder for dense embeddings
  - `mmarco-reranker/` - Cross-encoder for reranking
  - `opus-mt-en-de/` - Translation model for EN→DE query expansion

**kernel_output/:**
- Purpose: Outputs from Kaggle kernel runs (logs, intermediate files)
- Generated: Yes
- Committed: No

**.planning/codebase/:**
- Purpose: GSD-generated architecture and structure documentation
- Generated: Yes (by mapping process)
- Committed: Yes

## Key File Locations

**Entry Points:**
- `solution.py` - Main retrieval pipeline; execute with `python solution.py`; generates `submission.csv`
- `notebook_kaggle.ipynb` - Kaggle notebook version; cells run in order; same logic as solution.py
- `eval_local.py` - Quick validation using TF-IDF baseline; no model downloads needed

**Configuration:**
- `requirements.txt` - Python package dependencies (rank-bm25, transformers, torch, faiss-cpu, etc.)
- `SETUP.md` - How to set up environment, download models, run locally
- `kernel-metadata.json` - Kaggle notebook metadata (cell structure, dataset references)

**Core Logic:**
- `solution.py` - 445 lines; implements entire pipeline:
  - Lines 20-35: Config (Kaggle vs local path handling)
  - Lines 66-76: load_data()
  - Lines 82-114: BM25 building
  - Lines 120-166: Dense model loading and FAISS index building
  - Lines 172-189: Translation model loading
  - Lines 194-214: Reranker loading and scoring
  - Lines 220-282: RRF fusion and candidate retrieval
  - Lines 288-313: Evaluation metrics and calibration
  - Lines 319-330: Legal code extraction
  - Lines 336-440: Main execution loop

**Testing/Validation:**
- `eval_local.py` - 140 lines; TF-IDF baseline for quick validation
- `eval_dense.py` - Similar to eval_local but includes dense embeddings
- `diagnose.py` - Validate gold citation corpus coverage
- `diagnose2.py` through `diagnose6.py` - Test alternative retrieval strategies

**Utilities:**
- `download_models.py` - Download models from Hugging Face to local cache
- `watch_and_submit.sh` - Bash script to monitor file changes and submit to Kaggle
- `watch_simple.sh` - Simpler version of watch script

**Documentation:**
- `Overview.md` - Competition description and rules (from Kaggle)
- `AboutData.md` - Data format and field descriptions
- `SETUP.md` - Environment setup steps
- `CLAUDE.md` - Global Claude Code instructions (in parent directory)

## Naming Conventions

**Files:**
- Main logic: `solution.py`, `notebook_kaggle.ipynb` (descriptive names)
- Diagnostics: `diagnose.py`, `diagnose2.py`, ... (numbered variants for iteration)
- Evaluation: `eval_local.py`, `eval_dense.py` (eval prefix for metric scripts)
- Utilities: `download_models.py`, `watch_*.sh` (action verbs in names)
- Data: `train.csv`, `val.csv`, `test.csv`, `laws_de.csv`, `court_considerations.csv` (dataset name style)

**Directories:**
- Input: `Data/` (capitalized, singular for corpus)
- Models: `models/` (lowercase, plural for model cache)
- Output: `kernel_output/`, `.planning/` (functional names)
- Dotted: `.git/`, `.planning/` (Python/git convention for hidden dirs)

**Functions in solution.py:**
- camelCase verbs: `load_data()`, `build_bm25()`, `encode()`, `retrieve_candidates()`, `rrf_fuse()`
- noun phrases: `tokenize_for_bm25()`, `mean_pool()`, `extract_legal_codes()`
- adjective-noun: `macro_f1()`, `calibrate_top_k()`

**Variables:**
- snake_case: `laws_texts`, `q_emb`, `best_f1`, `val_results`, `rerank_tok`
- Config constants: UPPERCASE: `BM25_LAWS_K`, `RERANK_K`, `RRF_K_CONST`, `DEVICE`, `IS_KAGGLE`
- Loop indices: `i`, `j`, `rank` (single letter for iteration)
- Abbreviations preserved: `q_en`, `q_de`, `mt_tok`, `mt_mdl` (domain-specific abbreviations)

## Where to Add New Code

**New Feature (e.g., alternative reranker):**
- Primary code: `solution.py` - Add new `load_alternative_reranker()` function; add to main() orchestration
- Alternative: Create `eval_alternative.py` for isolated testing before integration
- Config: Add hyperparameter to top of solution.py (BM25_K, RERANK_K section)

**New Component/Module (e.g., new retrieval method):**
- Implementation: `solution.py` - Add `build_something_index()` and `retrieve_from_something()` functions
- Integration: Call from `retrieve_candidates()` and include in RRF fusion
- Testing: Create `diagnose_X.py` to test in isolation before full pipeline integration

**Utilities:**
- Shared helpers: `solution.py` - Keep all utility functions alongside main logic (no separate utils.py needed for this script size)
- Reusable evaluation: Create new `eval_X.py` file alongside `eval_local.py` for isolated testing

**Local Diagnostics:**
- Create new `diagnose_N.py` to experiment with variants (numbered pattern already established)
- Keep locally; do not commit unless it replaces or supercedes an existing diagnostic

## Special Directories

**Data/:**
- Purpose: Input CSV files (not code)
- Generated: No (provided by competition)
- Committed: No (too large; use gitignore or external storage)

**models/:**
- Purpose: Pretrained model weights cache
- Generated: Yes (by download_models.py or manual HF hub download)
- Committed: No (too large; ~880MB total)
- Note: In Kaggle environment, uploaded as separate datasets; not downloaded fresh in kernel

**kernel_output/:**
- Purpose: Log files and outputs from Kaggle kernel runs
- Generated: Yes (created by notebook execution)
- Committed: No (outputs only; not source)

**.planning/codebase/:**
- Purpose: GSD documentation (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by `/gsd-map-codebase` command)
- Committed: Yes (documentation is version controlled)

## Kaggle-Specific Structure

In Kaggle kernel environment, the structure differs:
```
/kaggle/
├── input/
│   ├── llm-agentic-legal-information-retrieval/  # Competition dataset (laws, court, queries)
│   ├── multilingual-e5-large/                     # Model dataset
│   ├── mmarco-reranker/                           # Model dataset
│   └── opus-mt-en-de/                             # Model dataset
└── output/
    └── submission.csv                             # Write output here
```

Paths in `solution.py` automatically detect Kaggle environment (`os.path.exists("/kaggle")`) and adjust paths accordingly (lines 20-35).

## Import Organization

**solution.py imports pattern:**
1. **Standard library** - os, pathlib, gc, re, time
2. **Numerical/data** - numpy, pandas, torch
3. **ML/NLP** - rank_bm25, transformers (AutoTokenizer, AutoModel, etc.), faiss
4. **Utility** - sklearn (for eval scripts only, not solution.py)

---

*Structure analysis: 2026-04-10*
