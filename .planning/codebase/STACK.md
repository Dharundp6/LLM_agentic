# Technology Stack

**Analysis Date:** 2026-04-10

## Languages

**Primary:**
- Python 3.x - Main implementation language for retrieval pipeline and model inference

**Secondary:**
- Bash - Shell scripts for automation (`watch_and_submit.sh`, `watch_simple.sh`)

## Runtime

**Environment:**
- Python runtime (local development or Kaggle kernel environment)

**Package Manager:**
- pip
- Lockfile: Not detected (uses `requirements.txt` instead)

## Frameworks

**Core ML/NLP:**
- transformers 4.38.0+ - Hugging Face library for pretrained model loading and inference
- torch (PyTorch) 2.1.0+ - Deep learning framework for model execution
- rank-bm25 0.2.2+ - BM25 lexical retrieval implementation

**Information Retrieval:**
- faiss-cpu 1.7.4+ - Dense vector similarity search (CPU version for Kaggle)
- scikit-learn - TF-IDF vectorization for fast lexical retrieval (`sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.linear_kernel`)

**Data Processing:**
- pandas 2.0.0+ - DataFrames for CSV handling and data manipulation
- numpy 1.24.0+ - Numerical computation and array operations

**Tokenization/NLP:**
- sentencepiece 0.1.99+ - Tokenizer used by multilingual models
- sacremoses 0.0.53+ - MOSES tokenizer for preprocessing

## Key Dependencies

**Critical:**
- transformers - Loads multilingual-e5-large, mmarco-mMiniLMv2-L12-H384-v1, and opus-mt-en-de models from Hugging Face Hub
- torch - Powers all neural model inference (dense embeddings, cross-encoder reranking)
- rank-bm25 - BM25Okapi implementation for lexical retrieval baseline (`solution.py` line 51)
- faiss - Dense vector indexing and search (referenced in `solution.py` line 57)
- scikit-learn - Fast TF-IDF baseline when GPU unavailable (`eval_local.py` line 10-11)

**Infrastructure:**
- pandas - Data loading and submission formatting
- numpy - Score aggregation and ranking operations
- huggingface-hub - Model downloading from HF Hub (`download_models.py` line 5)

## Configuration

**Environment:**
- Runtime detection: Kaggle vs. local via `os.path.exists("/kaggle")` check (`solution.py` lines 20-35)
- Conditional paths for data, models, and output based on execution context
- Device selection: Auto-detect CUDA availability (`DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")` in solution.py)

**Key Configurations:**
- BM25_LAWS_K = 100, BM25_COURT_K = 200, DENSE_LAWS_K = 100 - Retrieval candidate counts
- RERANK_K = 150 - Candidates sent to cross-encoder
- RRF_K_CONST = 60 - Reciprocal rank fusion constant for fusion of lexical + dense results

**Build:**
- No build system detected (pure Python)

## Platform Requirements

**Development:**
- Python 3.x with pip
- GPU optional (CPU fallback available but slow for dense embeddings)
- ~2GB+ RAM for loading laws corpus (~175K documents) and court corpus (~2.47M documents)
- Model disk space: ~880MB total for three models
  - multilingual-e5-large: ~560MB
  - mmarco-mMiniLMv2-L12-H384-v1: ~120MB
  - opus-mt-en-de: ~300MB

**Production (Kaggle):**
- Kaggle kernel with GPU accelerator (T4 x2 or P100 recommended)
- T4 GPU available in standard Kaggle notebooks
- Offline execution required - no internet access after kernel setup
- 12-hour execution limit (Kaggle notebook constraint)
- Models must be uploaded as Kaggle datasets for offline access

## Model Dependencies

**Downloaded from Hugging Face Hub:**
- `intfloat/multilingual-e5-large` - Dense multilingual retrieval encoder (560MB)
  - Usage: `AutoModel.from_pretrained()` in `notebook_kaggle.ipynb` cell 3
  - Tokenizer: `AutoTokenizer.from_pretrained()`
  - Device placement: CPU or CUDA depending on availability

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` - Multilingual cross-encoder reranker (120MB)
  - Usage: Reference in `solution.py` line 54 (`AutoModelForSequenceClassification`)
  - Purpose: Reranking retrieved candidate citations

- `Helsinki-NLP/opus-mt-en-de` - English-to-German translation (300MB)
  - Usage: `MarianMTModel` and `MarianTokenizer` in `solution.py` lines 55-56
  - Purpose: Query expansion via translation for BM25 improvement

---

*Stack analysis: 2026-04-10*
