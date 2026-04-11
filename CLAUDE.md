<!-- GSD:project-start source:PROJECT.md -->
## Project

**LLM Agentic Legal Information Retrieval**

A competitive Kaggle submission for the "LLM Agentic Legal Information Retrieval" competition. Given English legal questions about Swiss law, the system retrieves the most relevant Swiss legal citations (statutes and court decisions, mostly in German) and outputs them as submission.csv. The goal is to maximize Macro-F1 score and finish in the top 3 for prize money ($5K/$3K/$1K).

**Core Value:** Retrieve correct Swiss legal citations across language boundaries — every missed citation is a direct hit to F1 score.

### Constraints

- **Runtime:** 12-hour limit on Kaggle offline notebook — no internet access
- **Hardware:** T4 x2 (16GB VRAM each) — must fit Mistral-7B 4-bit + e5-large + cross-encoder
- **VRAM:** Careful model lifecycle management needed — load/unload models sequentially, not all at once
- **Corpus size:** 2.47M court decisions — dense encoding infeasible for full corpus, BM25 + sampled dense is the strategy
- **Code competition:** Must produce submission.csv reproducibly from the notebook
- **No external data:** Everything runs within Kaggle — no pre-uploaded datasets
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.x - Main implementation language for retrieval pipeline and model inference
- Bash - Shell scripts for automation (`watch_and_submit.sh`, `watch_simple.sh`)
## Runtime
- Python runtime (local development or Kaggle kernel environment)
- pip
- Lockfile: Not detected (uses `requirements.txt` instead)
## Frameworks
- transformers 4.38.0+ - Hugging Face library for pretrained model loading and inference
- torch (PyTorch) 2.1.0+ - Deep learning framework for model execution
- rank-bm25 0.2.2+ - BM25 lexical retrieval implementation
- faiss-cpu 1.7.4+ - Dense vector similarity search (CPU version for Kaggle)
- scikit-learn - TF-IDF vectorization for fast lexical retrieval (`sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.linear_kernel`)
- pandas 2.0.0+ - DataFrames for CSV handling and data manipulation
- numpy 1.24.0+ - Numerical computation and array operations
- sentencepiece 0.1.99+ - Tokenizer used by multilingual models
- sacremoses 0.0.53+ - MOSES tokenizer for preprocessing
## Key Dependencies
- transformers - Loads multilingual-e5-large, mmarco-mMiniLMv2-L12-H384-v1, and opus-mt-en-de models from Hugging Face Hub
- torch - Powers all neural model inference (dense embeddings, cross-encoder reranking)
- rank-bm25 - BM25Okapi implementation for lexical retrieval baseline (`solution.py` line 51)
- faiss - Dense vector indexing and search (referenced in `solution.py` line 57)
- scikit-learn - Fast TF-IDF baseline when GPU unavailable (`eval_local.py` line 10-11)
- pandas - Data loading and submission formatting
- numpy - Score aggregation and ranking operations
- huggingface-hub - Model downloading from HF Hub (`download_models.py` line 5)
## Configuration
- Runtime detection: Kaggle vs. local via `os.path.exists("/kaggle")` check (`solution.py` lines 20-35)
- Conditional paths for data, models, and output based on execution context
- Device selection: Auto-detect CUDA availability (`DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")` in solution.py)
- BM25_LAWS_K = 100, BM25_COURT_K = 200, DENSE_LAWS_K = 100 - Retrieval candidate counts
- RERANK_K = 150 - Candidates sent to cross-encoder
- RRF_K_CONST = 60 - Reciprocal rank fusion constant for fusion of lexical + dense results
- No build system detected (pure Python)
## Platform Requirements
- Python 3.x with pip
- GPU optional (CPU fallback available but slow for dense embeddings)
- ~2GB+ RAM for loading laws corpus (~175K documents) and court corpus (~2.47M documents)
- Model disk space: ~880MB total for three models
- Kaggle kernel with GPU accelerator (T4 x2 or P100 recommended)
- T4 GPU available in standard Kaggle notebooks
- Offline execution required - no internet access after kernel setup
- 12-hour execution limit (Kaggle notebook constraint)
- Models must be uploaded as Kaggle datasets for offline access
## Model Dependencies
- `intfloat/multilingual-e5-large` - Dense multilingual retrieval encoder (560MB)
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` - Multilingual cross-encoder reranker (120MB)
- `Helsinki-NLP/opus-mt-en-de` - English-to-German translation (300MB)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Lowercase with underscores: `solution.py`, `eval_local.py`, `diagnose.py`, `download_models.py`
- Diagnostic scripts: `diagnoseN.py` where N is an iteration number (e.g., `diagnose3.py`, `diagnose6.py`)
- Evaluation scripts: `eval_*.py` for different evaluation strategies
- Jupyter notebooks: `notebook_kaggle.ipynb`
- Lowercase with underscores for multi-word names: `load_data()`, `build_bm25()`, `encode()`, `mean_pool()`, `macro_f1()`
- Descriptive action verbs: `extract_legal_codes()`, `tokenize_for_bm25()`, `retrieve_candidates()`, `translate_batch()`
- Short utility functions: `log()`, `rrf()` (reciprocal rank fusion)
- Lowercase with underscores: `DATA_DIR`, `E5_DIR`, `RERANK_DIR`, `DEVICE`, `laws_texts`, `query_en`, `rerank_scores`
- All-caps for module-level constants: `IS_KAGGLE`, `BM25_LAWS_K`, `DENSE_LAWS_K`, `RERANK_K`, `RRF_K_CONST`
- Abbreviated model variables: `tok`, `mdl`, `enc`, `emb`, `gen` (transformer model loading context)
- Abbreviated data: `h` (hidden states), `m` (mask), `g` (gold set), `p` (predicted set), `c` (citation), `q` (query)
- Dictionary keys use underscores: `"query_id"`, `"gold_citations"`, `"predicted_citations"`, `"citation"`, `"text"`
- Set names ending in `_sets`: `gold_sets`, `pred_sets`, `all_cits`
- Index/ID naming: `doc_id`, `query_id`, `idx`
## Code Style
- No explicit formatter configuration (no `.pylintrc`, `.flake8`, or `black` config found)
- Observed style: 4-space indentation (Python standard)
- Line length: Generally under 100 characters; seen range 80-120 characters
- Multi-line function calls: Parameters aligned with opening parenthesis or indented consistently
- No linting configuration files present
- Code follows implicit Python conventions but not enforced by tooling
- No import aliases detected
- Full module paths used consistently: `from transformers import AutoTokenizer, AutoModel`
## Error Handling
- Assertions for validation: `assert DATA_DIR is not None, 'laws_de.csv not found'` (`notebook_kaggle.ipynb`)
- Minimal try/catch: Not observed in main code; assumes data files exist
- Early exits with `sys.exit()` not used; scripts silently fail if dependencies missing
- Optional recovery: Fallback citations used in `predict()` function when dense retrieval returns few candidates
- Descriptive and actionable: `'laws_de.csv not found'`, `'Model loaded. Params: {sum(p.numel() for p in mdl.parameters())/1e6:.0f}M'`
## Logging
- Explicitly flush to stdout for long-running processes: `print(msg, flush=True)` in `diagnose.py`, `diagnose3.py`
- Progress tracking in batch loops: `if i % 50000 == 0 and i > 0: print(f"  Dense encoded {i}/{len(texts)}")` (`solution.py`)
- Timing output: `print(f"  [{time.time()-t0:.1f}s]")` for elapsed time
- Section markers with `print("\n=== Section Name ===")` for readability
- Helper function `def log(msg): print(msg, flush=True)` used in evaluation scripts
- Info: `"Loading data..."`, `"BM25 laws: {len(laws_texts):,} docs [{time.time()-t0:.1f}s]"`
- Progress: `"  encoded {i}/{len(texts)}"`, `"  {row['query_id']}: gold={len(gold)} ..."`
- Results: `"*** TF-IDF baseline val macro-F1 = {best_f1:.4f} @ top-{best_k} ***"`
## Comments
- Strategy explanations at file level: Module docstrings explain retrieval approach, hyperparameters, and data flow
- Complex logic: `tokenize_for_bm25()` includes docstring explaining legal notation handling (`"Keep legal notation like 'Art.', 'Abs.', 'BGE', 'E.', numbers"`)
- Rationale for unusual choices: Comments on device selection (`"Force CPU — avoids P100 'no kernel image' error"`)
- Python docstrings used for function documentation
- Format: Triple-quoted strings with description and purpose
- Example from `solution.py`: `"""Returns list of (citation, full_text, rrf_score) tuples, sorted by RRF."""`
- Mostly absent; code is self-documenting with clear naming
- When used: E.g., `# Tokenize German query for BM25`, `# Dense on laws (use English query; multilingual-e5 handles cross-lingual)`
## Function Design
- Small utility functions: 3-15 lines (e.g., `mean_pool()`, `build_faiss_index()`)
- Medium functions: 15-40 lines (e.g., `encode()`, `retrieve_candidates()`)
- Larger functions: 40-80 lines (e.g., `run()`, `retrieve()` in notebook)
- Explicit keyword arguments for model tuning: `encode(tokenizer, model, texts, batch_size=64, prefix="passage:")`
- Dataframe row objects passed directly (not unpacked): `for _, row in val.iterrows()` then access via `row['column']`
- Batch processing in loops: `for i in range(0, len(texts), batch_size)`
- Tuples for related data: `(bm25_index, tokenized_texts)`, `(citation, score)`, `(scores.items())`
- Lists for sequences: `retrieve_candidates()` returns `list of (citation, text, score) tuples`
- NumPy arrays for embeddings: `encode()` returns `np.vstack(all_embs).astype("float32")`
- Dictionaries for structured results: `val_results.append({"query_id": ..., "gold": ..., "ranked_citations": ...})`
## Module Design
- No explicit `__all__` definitions
- Main executable: `if __name__ == "__main__": run()` pattern in `solution.py`
- Not used; monolithic script structure
- Configuration at module level: `IS_KAGGLE`, `DATA_DIR`, `E5_DIR`, `RERANK_DIR`, paths set once at top of file
- Device detection: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Model loading: Heavy models (E5, reranker) loaded in main flow, not in functions
## Type Hints
- Minimal type hints observed
- Function signatures mostly untyped: `def encode(tokenizer, model, texts, batch_size=64, prefix="passage:")` (no type annotations)
- Inferred from context: numpy arrays, pandas DataFrames, torch tensors understood implicitly
- NumPy arrays: `.astype("float32")` for embeddings, `.flatten()` for single query encoding
- PyTorch: `.to(DEVICE)`, `.eval()`, `.no_grad()` context for inference
## Data Handling
- Consistent column access: `row["query"]`, `row["gold_citations"]`, `row["citation"]`
- Iteration pattern: `for i, row in df.iterrows()` with explicit index tracking
- File I/O: `pd.read_csv(DATA_DIR / path)`, `df.to_csv(OUT_PATH, index=False)`
- Used for citation matching: `gold = set(row["gold_citations"].split(";"))`
- Set operations: intersection `&`, union `|` for evaluation
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- **Modular retrieval stages** - Each retrieval method (BM25, dense, translation) is independently loaded and executed
- **Fusion-based ranking** - Reciprocal Rank Fusion (RRF) combines multiple ranking signals before final reranking
- **Query expansion** - Queries are translated and augmented with extracted legal codes to improve cross-lingual and lexical matching
- **Calibration-driven output** - Citation count per query is optimized on validation set using macro-F1 metric
- **Staged inference optimization** - Models are loaded once, then applied to batches to minimize memory footprint
## Layers
- Purpose: Load corpus (laws and court decisions) and queries; validate gold citations exist in corpus
- Location: `load_data()` in `solution.py` (lines 66-76); diagnostic scripts in `diagnose.py`, `diagnose2.py`, etc.
- Contains: CSV parsing, data shape inspection, citation validation
- Depends on: pandas, pathlib for file I/O
- Used by: All retrieval components
- Purpose: Fast, query-independent lexical matching using BM25 scoring
- Location: `build_bm25_indices()` in `solution.py` (lines 100-114); `tokenize_for_bm25()` (lines 82-89)
- Contains: BM25Okapi tokenizers for laws and court corpus, custom regex-based tokenization preserving legal notation
- Depends on: rank-bm25, numpy
- Used by: Candidate retrieval and RRF fusion
- Purpose: Cross-lingual semantic matching using multilingual-e5-large embeddings
- Location: `load_dense_model()` (lines 120-124), `encode()` (lines 132-145), `build_dense_index()` (lines 155-166) in `solution.py`
- Contains: Hugging Face model loading, batch-wise embedding generation with mean pooling, L2 normalization, FAISS index building
- Depends on: transformers, torch, faiss, numpy
- Used by: Dense candidate retrieval for laws corpus only
- Purpose: Translate English queries to German; extract legal codes for improved BM25 matching
- Location: `load_translation_model()` (lines 172-176), `translate_batch()` (lines 179-188), `extract_legal_codes()` (lines 319-330) in `solution.py`
- Contains: MarianMT translation model, batch translation, regex-based legal code extraction (e.g., 'Art. 221 StPO')
- Depends on: transformers, torch, regex
- Used by: Retrieval candidates function
- Purpose: Orchestrate lexical + dense retrieval and combine via RRF
- Location: `retrieve_candidates()` (lines 229-282), `rrf_fuse()` (lines 220-226) in `solution.py`
- Contains: BM25 scoring on German tokens, FAISS kNN search, RRF fusion algorithm
- Depends on: BM25, FAISS indices, dense model, translation
- Used by: Reranking pipeline
- Purpose: Final ranking of candidates using multilingual mMiniLM cross-encoder
- Location: `load_reranker()` (lines 194-198), `rerank()` (lines 201-214) in `solution.py`
- Contains: Sequence classification model, pairwise scoring of query vs candidate pairs
- Depends on: transformers, torch, numpy
- Used by: Final prediction ranking
- Purpose: Compute macro-F1 metric; optimize citation count threshold on validation set
- Location: `macro_f1()` (lines 288-300), `calibrate_top_k()` (lines 303-313) in `solution.py`
- Contains: Per-query F1 computation, greedy grid search over top-K values (1-80)
- Depends on: numpy
- Used by: Test set prediction count determination
## Data Flow
- **Indices**: BM25Okapi objects (immutable, query-independent)
- **Embeddings**: FAISS IndexFlatIP (immutable; precomputed, not query-dependent)
- **Models**: Loaded once per run, then reused for batches to minimize memory
- **Results**: Accumulate in `val_results` list during validation; format to DataFrame for submission CSV
## Key Abstractions
- Purpose: Preserve legal notation (Art., abbreviations) while lowercasing and splitting
- Location: `tokenize_for_bm25()` in `solution.py` lines 82-89
- Pattern: Regex `r"[\w.]+"` to keep periods (legal convention), lowercase for case-insensitivity
- Used by: BM25 index building and scoring
- Purpose: Unsupervised fusion of ranked lists from BM25 (laws), BM25 (court), and dense retrieval
- Location: `rrf_fuse()` in `solution.py` lines 220-226
- Pattern: Score = sum(1 / (k + rank + 1)) for each list; k=60 (tunable constant)
- Used by: Combining heterogeneous retrieval signals
- Purpose: Regex-based extraction of Swiss law abbreviations and article numbers
- Location: `extract_legal_codes()` in `solution.py` lines 319-330
- Pattern: Predefined set of law codes (ZGB, OR, StGB, StPO, BGG, etc.); extract "Art. N" patterns
- Used by: Query expansion for German BM25
- Purpose: Per-query F1 averaging (not per-citation microaverage)
- Location: `macro_f1()` in `solution.py` lines 288-300
- Pattern: Compute TP/precision/recall per query; return mean across queries; handle empty sets
- Used by: Validation evaluation and top-K calibration
- Purpose: Convert transformer outputs to fixed-size normalized embeddings
- Location: `mean_pool()` in `solution.py` lines 127-129
- Pattern: Apply attention mask as weights; L2 normalize to unit sphere for cosine similarity in FAISS
- Used by: Dense embedding generation
## Entry Points
- Location: `solution.py` (lines 443-444)
- Triggers: `python solution.py` or Kaggle notebook cell execution
- Responsibilities:
- Location: `diagnose.py`, `diagnose2.py`, ..., `diagnose6.py`
- Triggers: Manual execution for debugging specific components
- Responsibilities: Validate corpus, test retrieval strategies, profile performance
- Location: `eval_local.py`, `eval_dense.py`
- Triggers: Quick local validation before Kaggle submission
- Responsibilities: Fast TF-IDF baseline, dense embedding pipeline validation
- Location: `notebook_kaggle.ipynb`
- Triggers: Kaggle kernel execution
- Responsibilities: Same as solution.py but organized as cells for interactive development
## Error Handling
- **Missing data**: `fillna("")` for CSV columns (assumes all citations exist in corpus if in gold set)
- **Empty candidates**: Check `if candidates:` before reranking; return empty prediction if no matches
- **Batch processing**: No explicit error handling; if a model fails, entire query fails
- **CUDA fallback**: Auto-detect CUDA; fall back to CPU if not available (`torch.device(...cuda... else cpu)`)
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
