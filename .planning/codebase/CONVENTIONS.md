# Coding Conventions

**Analysis Date:** 2026-04-10

## Naming Patterns

**Files:**
- Lowercase with underscores: `solution.py`, `eval_local.py`, `diagnose.py`, `download_models.py`
- Diagnostic scripts: `diagnoseN.py` where N is an iteration number (e.g., `diagnose3.py`, `diagnose6.py`)
- Evaluation scripts: `eval_*.py` for different evaluation strategies
- Jupyter notebooks: `notebook_kaggle.ipynb`

**Functions:**
- Lowercase with underscores for multi-word names: `load_data()`, `build_bm25()`, `encode()`, `mean_pool()`, `macro_f1()`
- Descriptive action verbs: `extract_legal_codes()`, `tokenize_for_bm25()`, `retrieve_candidates()`, `translate_batch()`
- Short utility functions: `log()`, `rrf()` (reciprocal rank fusion)

**Variables:**
- Lowercase with underscores: `DATA_DIR`, `E5_DIR`, `RERANK_DIR`, `DEVICE`, `laws_texts`, `query_en`, `rerank_scores`
- All-caps for module-level constants: `IS_KAGGLE`, `BM25_LAWS_K`, `DENSE_LAWS_K`, `RERANK_K`, `RRF_K_CONST`
- Abbreviated model variables: `tok`, `mdl`, `enc`, `emb`, `gen` (transformer model loading context)
- Abbreviated data: `h` (hidden states), `m` (mask), `g` (gold set), `p` (predicted set), `c` (citation), `q` (query)

**Types:**
- Dictionary keys use underscores: `"query_id"`, `"gold_citations"`, `"predicted_citations"`, `"citation"`, `"text"`
- Set names ending in `_sets`: `gold_sets`, `pred_sets`, `all_cits`
- Index/ID naming: `doc_id`, `query_id`, `idx`

## Code Style

**Formatting:**
- No explicit formatter configuration (no `.pylintrc`, `.flake8`, or `black` config found)
- Observed style: 4-space indentation (Python standard)
- Line length: Generally under 100 characters; seen range 80-120 characters
- Multi-line function calls: Parameters aligned with opening parenthesis or indented consistently

**Linting:**
- No linting configuration files present
- Code follows implicit Python conventions but not enforced by tooling

**Import Organization:**

Order observed in `solution.py`:
1. Standard library: `os`, `gc`, `re`, `time`
2. Data science libraries: `numpy`, `pandas`, `torch`
3. Specialized packages: `rank_bm25`, `transformers`, `faiss`

Order observed in Jupyter notebook:
1. OS/utility: `os`, `gc`, `re`, `time`
2. Data science: `numpy`, `pandas`, `torch`
3. Path handling: `pathlib.Path`
4. Utilities: `collections.Counter`
5. HuggingFace: `transformers.AutoTokenizer`, `transformers.AutoModel`

**Path Aliases:**
- No import aliases detected
- Full module paths used consistently: `from transformers import AutoTokenizer, AutoModel`

## Error Handling

**Patterns:**
- Assertions for validation: `assert DATA_DIR is not None, 'laws_de.csv not found'` (`notebook_kaggle.ipynb`)
- Minimal try/catch: Not observed in main code; assumes data files exist
- Early exits with `sys.exit()` not used; scripts silently fail if dependencies missing
- Optional recovery: Fallback citations used in `predict()` function when dense retrieval returns few candidates

**Error Messages:**
- Descriptive and actionable: `'laws_de.csv not found'`, `'Model loaded. Params: {sum(p.numel() for p in mdl.parameters())/1e6:.0f}M'`

## Logging

**Framework:** `print()` with flush flag

**Patterns:**
- Explicitly flush to stdout for long-running processes: `print(msg, flush=True)` in `diagnose.py`, `diagnose3.py`
- Progress tracking in batch loops: `if i % 50000 == 0 and i > 0: print(f"  Dense encoded {i}/{len(texts)}")` (`solution.py`)
- Timing output: `print(f"  [{time.time()-t0:.1f}s]")` for elapsed time
- Section markers with `print("\n=== Section Name ===")` for readability
- Helper function `def log(msg): print(msg, flush=True)` used in evaluation scripts

**Log Level Examples:**
- Info: `"Loading data..."`, `"BM25 laws: {len(laws_texts):,} docs [{time.time()-t0:.1f}s]"`
- Progress: `"  encoded {i}/{len(texts)}"`, `"  {row['query_id']}: gold={len(gold)} ..."`
- Results: `"*** TF-IDF baseline val macro-F1 = {best_f1:.4f} @ top-{best_k} ***"`

## Comments

**When to Comment:**
- Strategy explanations at file level: Module docstrings explain retrieval approach, hyperparameters, and data flow
- Complex logic: `tokenize_for_bm25()` includes docstring explaining legal notation handling (`"Keep legal notation like 'Art.', 'Abs.', 'BGE', 'E.', numbers"`)
- Rationale for unusual choices: Comments on device selection (`"Force CPU — avoids P100 'no kernel image' error"`)

**JSDoc/TSDoc:**
- Python docstrings used for function documentation
- Format: Triple-quoted strings with description and purpose
- Example from `solution.py`: `"""Returns list of (citation, full_text, rrf_score) tuples, sorted by RRF."""`

**Minimal Inline Comments:**
- Mostly absent; code is self-documenting with clear naming
- When used: E.g., `# Tokenize German query for BM25`, `# Dense on laws (use English query; multilingual-e5 handles cross-lingual)`

## Function Design

**Size:** 
- Small utility functions: 3-15 lines (e.g., `mean_pool()`, `build_faiss_index()`)
- Medium functions: 15-40 lines (e.g., `encode()`, `retrieve_candidates()`)
- Larger functions: 40-80 lines (e.g., `run()`, `retrieve()` in notebook)

**Parameters:** 
- Explicit keyword arguments for model tuning: `encode(tokenizer, model, texts, batch_size=64, prefix="passage:")`
- Dataframe row objects passed directly (not unpacked): `for _, row in val.iterrows()` then access via `row['column']`
- Batch processing in loops: `for i in range(0, len(texts), batch_size)`

**Return Values:**
- Tuples for related data: `(bm25_index, tokenized_texts)`, `(citation, score)`, `(scores.items())`
- Lists for sequences: `retrieve_candidates()` returns `list of (citation, text, score) tuples`
- NumPy arrays for embeddings: `encode()` returns `np.vstack(all_embs).astype("float32")`
- Dictionaries for structured results: `val_results.append({"query_id": ..., "gold": ..., "ranked_citations": ...})`

## Module Design

**Exports:**
- No explicit `__all__` definitions
- Main executable: `if __name__ == "__main__": run()` pattern in `solution.py`

**Barrel Files:**
- Not used; monolithic script structure

**Global State:**
- Configuration at module level: `IS_KAGGLE`, `DATA_DIR`, `E5_DIR`, `RERANK_DIR`, paths set once at top of file
- Device detection: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Model loading: Heavy models (E5, reranker) loaded in main flow, not in functions

## Type Hints

**Usage:**
- Minimal type hints observed
- Function signatures mostly untyped: `def encode(tokenizer, model, texts, batch_size=64, prefix="passage:")` (no type annotations)
- Inferred from context: numpy arrays, pandas DataFrames, torch tensors understood implicitly

**NumPy/Torch Conventions:**
- NumPy arrays: `.astype("float32")` for embeddings, `.flatten()` for single query encoding
- PyTorch: `.to(DEVICE)`, `.eval()`, `.no_grad()` context for inference

## Data Handling

**DataFrames:**
- Consistent column access: `row["query"]`, `row["gold_citations"]`, `row["citation"]`
- Iteration pattern: `for i, row in df.iterrows()` with explicit index tracking
- File I/O: `pd.read_csv(DATA_DIR / path)`, `df.to_csv(OUT_PATH, index=False)`

**Sets:**
- Used for citation matching: `gold = set(row["gold_citations"].split(";"))`
- Set operations: intersection `&`, union `|` for evaluation

---

*Convention analysis: 2026-04-10*
