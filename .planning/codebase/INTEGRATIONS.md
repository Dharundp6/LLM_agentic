# External Integrations

**Analysis Date:** 2026-04-10

## APIs & External Services

**Hugging Face Hub:**
- Model repository for pretrained model downloads
  - SDK/Client: `huggingface_hub.snapshot_download()` and `transformers.AutoModel/AutoTokenizer`
  - Models fetched: multilingual-e5-large, mmarco-mMiniLMv2-L12-H384-v1, Helsinki-NLP/opus-mt-en-de
  - Authentication: HF token (implicit if `huggingface-hub` CLI configured, optional for public models)
  - Usage locations: `download_models.py` (line 49-50), `solution.py` (lines 52-56), `notebook_kaggle.ipynb` (cell 3)

**Kaggle API:**
- Dataset upload and notebook submission
  - SDK/Client: `kaggle` CLI tool
  - Auth: `kaggle.json` file (credentials for Kaggle authentication)
  - Usage: `watch_and_submit.sh` for automated notebook submission
  - File location: `kaggle.json` in project root (not read - credential file)

**Kaggle Kernel Environment:**
- Execution environment for competition submission
  - Input datasets mounted at: `/kaggle/input/`
  - Output path: `submission.csv` written to kernel working directory
  - Auto-detected in code: `IS_KAGGLE = os.path.exists("/kaggle")` (`solution.py` line 21)
  - Constraints: 12-hour limit, offline execution (no internet after kernel setup), GPU optional (T4/P100)

## Data Storage

**Databases:**
- Not applicable - no persistent database

**File Storage:**
- Local filesystem only (CSV-based)
  - Locations: `Data/` directory containing:
    - `laws_de.csv` - Swiss federal laws corpus (~175K rows, German text)
    - `court_considerations.csv` - Swiss federal court decision corpus (~2.47M rows, German text)
    - `train.csv` - Training queries with gold citations
    - `val.csv` - Validation queries with gold citations
    - `test.csv` - Test queries (no gold citations)
    - `sample_submission.csv` - Submission format template

- Model storage:
  - Kaggle: models uploaded as datasets to `/kaggle/input/`
  - Local dev: Models cached in `models/` directory or downloaded on-demand from HF Hub
  - Paths configured in `solution.py` lines 24-35

**Caching:**
- Transformers library cache: Default ~/.cache/huggingface/ for downloaded models
- FAISS indices: Built in-memory during notebook execution, not persisted
- TF-IDF vectorizers: Fit and stored in memory during pipeline execution

## Authentication & Identity

**Auth Provider:**
- None for the retrieval system itself (stateless inference)

**External Auth Requirements:**
- Hugging Face: Optional token for private models (not used - all models are public)
- Kaggle: API key via `kaggle.json` for dataset/notebook operations
- No user authentication layer in the retrieval pipeline

## Monitoring & Observability

**Error Tracking:**
- Not integrated - errors logged to console only

**Logs:**
- Console output only (print statements)
- Structured logging via `print()` with timestamps (`time.time()`)
- Performance metrics: Retrieval times, F1 scores, memory usage printed to stdout
- Examples in `solution.py` (lines 73-75), `notebook_kaggle.ipynb` (cells 1-7)

**Metrics:**
- Macro F1 score computed per-query (defined in `diagnose6.py` lines 26-34)
- Validation metrics tracked in `notebook_kaggle.ipynb` cell 6
- No external metrics collection service

## CI/CD & Deployment

**Hosting:**
- Kaggle (code competition platform)
- Notebook submission required for official results

**CI Pipeline:**
- None detected

**Local Execution Path:**
1. `python eval_local.py` - Quick BM25-only baseline validation
2. `python download_models.py` - Download and cache models locally
3. `python solution.py` - Full pipeline (if models available locally)

**Kaggle Execution Path:**
1. Upload notebook (`notebook_kaggle.ipynb`) to Kaggle
2. Add datasets: competition data + model datasets
3. Enable GPU accelerator (T4 x2 or P100)
4. Disable internet access (required)
5. Run notebook → generates `submission.csv`
6. Download submission for leaderboard upload

## Environment Configuration

**Required env vars:**
- None explicitly defined (paths hardcoded based on execution context)

**Secrets location:**
- `kaggle.json` - API credentials for Kaggle CLI (not read - credential file)

**Path Configuration (context-aware):**
```python
# From solution.py lines 20-35
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    DATA_DIR   = Path("/kaggle/input/llm-agentic-legal-information-retrieval")
    E5_DIR     = Path("/kaggle/input/multilingual-e5-large")
    RERANK_DIR = Path("/kaggle/input/mmarco-reranker")
    OPUSMT_DIR = Path("/kaggle/input/opus-mt-en-de")
    OUT_PATH   = Path("submission.csv")
else:
    # Local paths for development
    DATA_DIR   = Path(r"C:\Users\Dharun prasanth\OneDrive\Documents\Projects\LLm_Agentic\Data")
    E5_DIR     = "intfloat/multilingual-e5-large"  # HF hub ID
    RERANK_DIR = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    OPUSMT_DIR = "Helsinki-NLP/opus-mt-en-de"
    OUT_PATH   = Path(r"C:\Users\Dharun prasanth\OneDrive\Documents\Projects\LLm_Agentic\submission.csv")
```

## Webhooks & Callbacks

**Incoming:**
- Not applicable

**Outgoing:**
- Kaggle dataset/notebook API calls for submission (via `kaggle` CLI)
- No other external webhooks

## Model Serving

**Inference Architecture:**
- Local in-process inference (no API server)
- Models loaded via transformers library: `AutoModel.from_pretrained()`, `AutoModelForSequenceClassification.from_pretrained()`
- Batch processing for efficiency (documented in `notebook_kaggle.ipynb` cell 3: batch_size=64)
- Device selection: CPU or CUDA auto-detected (forced CPU in some notebook versions to avoid P100 compatibility issues)

**Model Pipeline:**
1. **Encoding**: Query and corpus encoding with multilingual-e5-large (mean pooling + L2 normalization)
2. **Retrieval**: Dense vector similarity search with FAISS, lexical search with BM25, fusion with Reciprocal Rank Fusion (RRF)
3. **Reranking**: Cross-encoder scoring with mmarco-mMiniLMv2-L12-H384-v1 (optional in full pipeline)
4. **Post-processing**: Citation deduplication, frequency-based fallback from training set

**Kaggle Dataset Dependencies:**
- `multilingual-e5-large` - Must be uploaded as Kaggle dataset
- `mmarco-reranker` - Must be uploaded as Kaggle dataset
- `opus-mt-en-de` - Must be uploaded as Kaggle dataset
- Competition data - Auto-added by Kaggle

---

*Integration audit: 2026-04-10*
