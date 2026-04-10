# Architecture

**Analysis Date:** 2026-04-10

## Pattern Overview

**Overall:** Hybrid Information Retrieval Pipeline with Multi-Stage Ranking

This is a **multi-stage legal document retrieval system** that combines lexical (BM25), dense semantic (neural embeddings), and learned-to-rank (cross-encoder) approaches. The system is designed for the Swiss legal domain where queries come in English but must retrieve documents in German.

**Key Characteristics:**
- **Modular retrieval stages** - Each retrieval method (BM25, dense, translation) is independently loaded and executed
- **Fusion-based ranking** - Reciprocal Rank Fusion (RRF) combines multiple ranking signals before final reranking
- **Query expansion** - Queries are translated and augmented with extracted legal codes to improve cross-lingual and lexical matching
- **Calibration-driven output** - Citation count per query is optimized on validation set using macro-F1 metric
- **Staged inference optimization** - Models are loaded once, then applied to batches to minimize memory footprint

## Layers

**Data Loading & Preprocessing:**
- Purpose: Load corpus (laws and court decisions) and queries; validate gold citations exist in corpus
- Location: `load_data()` in `solution.py` (lines 66-76); diagnostic scripts in `diagnose.py`, `diagnose2.py`, etc.
- Contains: CSV parsing, data shape inspection, citation validation
- Depends on: pandas, pathlib for file I/O
- Used by: All retrieval components

**Lexical Retrieval (BM25):**
- Purpose: Fast, query-independent lexical matching using BM25 scoring
- Location: `build_bm25_indices()` in `solution.py` (lines 100-114); `tokenize_for_bm25()` (lines 82-89)
- Contains: BM25Okapi tokenizers for laws and court corpus, custom regex-based tokenization preserving legal notation
- Depends on: rank-bm25, numpy
- Used by: Candidate retrieval and RRF fusion

**Dense Semantic Retrieval (Neural Embeddings):**
- Purpose: Cross-lingual semantic matching using multilingual-e5-large embeddings
- Location: `load_dense_model()` (lines 120-124), `encode()` (lines 132-145), `build_dense_index()` (lines 155-166) in `solution.py`
- Contains: Hugging Face model loading, batch-wise embedding generation with mean pooling, L2 normalization, FAISS index building
- Depends on: transformers, torch, faiss, numpy
- Used by: Dense candidate retrieval for laws corpus only

**Query Processing & Expansion:**
- Purpose: Translate English queries to German; extract legal codes for improved BM25 matching
- Location: `load_translation_model()` (lines 172-176), `translate_batch()` (lines 179-188), `extract_legal_codes()` (lines 319-330) in `solution.py`
- Contains: MarianMT translation model, batch translation, regex-based legal code extraction (e.g., 'Art. 221 StPO')
- Depends on: transformers, torch, regex
- Used by: Retrieval candidates function

**Candidate Retrieval & Fusion:**
- Purpose: Orchestrate lexical + dense retrieval and combine via RRF
- Location: `retrieve_candidates()` (lines 229-282), `rrf_fuse()` (lines 220-226) in `solution.py`
- Contains: BM25 scoring on German tokens, FAISS kNN search, RRF fusion algorithm
- Depends on: BM25, FAISS indices, dense model, translation
- Used by: Reranking pipeline

**Cross-Encoder Reranking:**
- Purpose: Final ranking of candidates using multilingual mMiniLM cross-encoder
- Location: `load_reranker()` (lines 194-198), `rerank()` (lines 201-214) in `solution.py`
- Contains: Sequence classification model, pairwise scoring of query vs candidate pairs
- Depends on: transformers, torch, numpy
- Used by: Final prediction ranking

**Evaluation & Calibration:**
- Purpose: Compute macro-F1 metric; optimize citation count threshold on validation set
- Location: `macro_f1()` (lines 288-300), `calibrate_top_k()` (lines 303-313) in `solution.py`
- Contains: Per-query F1 computation, greedy grid search over top-K values (1-80)
- Depends on: numpy
- Used by: Test set prediction count determination

## Data Flow

**Retrieval Pipeline (Main):**

1. **Load Stage** - Read laws.csv, court_considerations.csv, val/test queries from CSV files
2. **Index Building** - Construct BM25 indices for laws and court; build FAISS index for dense laws embeddings
3. **Model Loading** - Load dense encoder (multilingual-e5-large), translator (opus-mt-en-de), reranker (mmarco)
4. **Query Processing** - For each query:
   - Translate English query to German via MarianMT
   - Extract legal codes (Art. numbers, law abbreviations) from original English query
   - Append codes to translated query for BM25 signal boost
5. **Candidate Retrieval** - For expanded German query:
   - BM25 on laws: retrieve top-100 document IDs
   - BM25 on court: retrieve top-200 document IDs
   - Dense (FAISS) on laws: retrieve top-100 embeddings using English query
   - RRF fusion of three ranked lists
   - Dedup and collect top-150 candidates
6. **Reranking** - Cross-encoder scores top-150 candidates against original English query
7. **Threshold Selection** - On validation set: grid search for top-K that maximizes macro-F1 (range 1-80)
8. **Test Prediction** - Apply best top-K to test queries; format citations as submission CSV

**Evaluation Loop:**
1. Process all validation queries through full pipeline
2. Collect ranked citation predictions
3. Compute macro-F1 for each top-K threshold
4. Select K with highest F1
5. Log best F1 and K value for reproducibility

**State Management:**
- **Indices**: BM25Okapi objects (immutable, query-independent)
- **Embeddings**: FAISS IndexFlatIP (immutable; precomputed, not query-dependent)
- **Models**: Loaded once per run, then reused for batches to minimize memory
- **Results**: Accumulate in `val_results` list during validation; format to DataFrame for submission CSV

## Key Abstractions

**BM25 Tokenizer:**
- Purpose: Preserve legal notation (Art., abbreviations) while lowercasing and splitting
- Location: `tokenize_for_bm25()` in `solution.py` lines 82-89
- Pattern: Regex `r"[\w.]+"` to keep periods (legal convention), lowercase for case-insensitivity
- Used by: BM25 index building and scoring

**Reciprocal Rank Fusion (RRF):**
- Purpose: Unsupervised fusion of ranked lists from BM25 (laws), BM25 (court), and dense retrieval
- Location: `rrf_fuse()` in `solution.py` lines 220-226
- Pattern: Score = sum(1 / (k + rank + 1)) for each list; k=60 (tunable constant)
- Used by: Combining heterogeneous retrieval signals

**Legal Code Extraction:**
- Purpose: Regex-based extraction of Swiss law abbreviations and article numbers
- Location: `extract_legal_codes()` in `solution.py` lines 319-330
- Pattern: Predefined set of law codes (ZGB, OR, StGB, StPO, BGG, etc.); extract "Art. N" patterns
- Used by: Query expansion for German BM25

**Macro-F1 Metric:**
- Purpose: Per-query F1 averaging (not per-citation microaverage)
- Location: `macro_f1()` in `solution.py` lines 288-300
- Pattern: Compute TP/precision/recall per query; return mean across queries; handle empty sets
- Used by: Validation evaluation and top-K calibration

**Mean Pooling Normalization:**
- Purpose: Convert transformer outputs to fixed-size normalized embeddings
- Location: `mean_pool()` in `solution.py` lines 127-129
- Pattern: Apply attention mask as weights; L2 normalize to unit sphere for cosine similarity in FAISS
- Used by: Dense embedding generation

## Entry Points

**Main Script:**
- Location: `solution.py` (lines 443-444)
- Triggers: `python solution.py` or Kaggle notebook cell execution
- Responsibilities:
  - Load all data (laws, court, val, test)
  - Build BM25 and FAISS indices
  - Load transformer models
  - Process validation queries; calibrate top-K
  - Process test queries; save submission.csv

**Diagnostic Scripts:**
- Location: `diagnose.py`, `diagnose2.py`, ..., `diagnose6.py`
- Triggers: Manual execution for debugging specific components
- Responsibilities: Validate corpus, test retrieval strategies, profile performance

**Evaluation Scripts:**
- Location: `eval_local.py`, `eval_dense.py`
- Triggers: Quick local validation before Kaggle submission
- Responsibilities: Fast TF-IDF baseline, dense embedding pipeline validation

**Jupyter Notebook:**
- Location: `notebook_kaggle.ipynb`
- Triggers: Kaggle kernel execution
- Responsibilities: Same as solution.py but organized as cells for interactive development

## Error Handling

**Strategy:** Minimal error handling; fail-fast approach suitable for deterministic processing

**Patterns:**
- **Missing data**: `fillna("")` for CSV columns (assumes all citations exist in corpus if in gold set)
- **Empty candidates**: Check `if candidates:` before reranking; return empty prediction if no matches
- **Batch processing**: No explicit error handling; if a model fails, entire query fails
- **CUDA fallback**: Auto-detect CUDA; fall back to CPU if not available (`torch.device(...cuda... else cpu)`)

## Cross-Cutting Concerns

**Logging:** Simple `print(..., flush=True)` statements for pipeline progress; timing via `time.time()`

**Validation:** Corpus validation via gold citation lookup (`diagnose.py`); F1 macro-averaging confirms metric alignment

**Authentication:** No auth required; models loaded from Hugging Face Hub (local dev) or Kaggle datasets (offline)

**Memory Management:** Explicit `gc.collect()` after dense embedding to reduce peaks; batch processing to avoid loading all candidates at once

**Device Management:** CUDA auto-detection; all models moved to DEVICE via `.to(DEVICE)`; embeddings moved back to CPU for FAISS compatibility

---

*Architecture analysis: 2026-04-10*
