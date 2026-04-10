# Technology Stack — LLM Agentic Legal Information Retrieval

**Project:** Kaggle cross-lingual legal citation retrieval (en→de, Swiss law)
**Researched:** 2026-04-10
**Overall Confidence:** MEDIUM-HIGH (GPU/VRAM figures are empirically sourced; legal-domain benchmarks are sparse)

---

## Recommended Stack

### 1. Multilingual Dense Embedding Model

| Recommendation | Model | Size | VRAM (fp16) |
|----------------|-------|------|-------------|
| PRIMARY | `BAAI/bge-m3` | ~1.1 GB disk / ~560M params | ~4–6 GB at batch_size=8, max_length=512 |
| FALLBACK | `intfloat/multilingual-e5-large` | ~560 MB / ~560M params | ~3–4 GB |

**Why BGE-M3 over multilingual-e5-large:**

- BGE-M3 delivers dense + sparse + ColBERT retrieval from a single model. For the laws corpus (175K docs), you can run all three modes and fuse them internally — eliminating the need for a separate BM25 index on that corpus entirely.
- On MIRACL (18 languages, cross-lingual retrieval benchmark), BGE-M3 dense alone scores nDCG@10 ~65+, while all-modes combined reaches ~70, surpassing mE5-large (~65.4) by a meaningful margin.
- BGE-M3 supports 8192-token sequences (critical for verbose Swiss legal statute text) versus mE5-large's 512 tokens.
- For German-specific cross-lingual retrieval (en query → de document), BGE-M3's training on MKQA and 100+ languages gives it direct advantage.

**Why not Qwen3-Embedding-8B (ranked #1 MMTEB 2025):**

- 8B parameters require ~16 GB VRAM in fp16. With two T4 GPUs (16 GB each), loading an 8B embedder alongside a 7B LLM and a cross-encoder simultaneously is infeasible. Sequential loading adds ~10–20 minutes per query batch and blows the 12-hour runtime budget.
- BGE-M3 at 560M params leaves headroom for the reranker and LLM.

**Usage notes:**

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
# For laws corpus (dense + sparse unified):
output = model.encode(passages, return_dense=True, return_sparse=True, return_colbert_vecs=False, max_length=512)
# Cross-lingual: encode English query, German passages — no instruction prefix needed
```

**Confidence:** HIGH (MIRACL benchmarks, official HF model card, community VRAM reports)

---

### 2. BM25 / Sparse Retrieval Library

| Recommendation | Library | Version |
|----------------|---------|---------|
| PRIMARY | `bm25s` | 0.2.x (PyPI) |
| WHAT WE HAVE | `rank-bm25` | 0.2.2 |

**Why bm25s over rank-bm25:**

- rank-bm25 uses pure Python loops — it is prohibitively slow at 2.47M documents. Indexing 2.47M docs with rank-bm25 at scale takes 30–60+ minutes and consumes >8 GB RAM.
- bm25s uses eager sparse scoring with Scipy sparse matrices, achieving **500x speedup** over rank-bm25 in benchmarks. The library was specifically validated on Natural Questions (2M+ documents) and can index that corpus in minutes.
- bm25s supports memory-mapped indices: build once, save to disk, reload fast — critical for Kaggle's offline constraint where you want to avoid rebuilding on each run.
- bm25s has no Java dependency (unlike Elasticsearch/Lucene) and no server setup — installs with `pip install bm25s`.

**What NOT to use:**

- `rank-bm25` — too slow for 2.47M docs; acceptable only for the 175K laws corpus as a quick fallback.
- Elasticsearch — requires a running Java server; not feasible in a Kaggle offline notebook.
- `retriv` — more performant than rank-bm25 but adds a dependency layer; bm25s is simpler and faster.

**Installation:**

```bash
pip install bm25s[full]   # includes Numba acceleration
```

**Usage pattern for court corpus (2.47M docs):**

```python
import bm25s, numpy as np
corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="de")  # German stopwords
retriever = bm25s.BM25()
retriever.index(corpus_tokens)
retriever.save("bm25_court_index")   # persist to disk
# Query: translate English query to German first, then:
results, scores = retriever.retrieve(bm25s.tokenize([german_query]), k=200)
```

**Confidence:** HIGH (bm25s paper, PyPI benchmarks, official 2M-doc example confirmed)

---

### 3. Cross-Encoder Reranker

| Recommendation | Model | Params | VRAM |
|----------------|-------|--------|------|
| PRIMARY | `jinaai/jina-reranker-v2-base-multilingual` | 278M | ~2–3 GB fp16 |
| SECONDARY | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | ~120M | ~1 GB (current) |

**Why jina-reranker-v2 over mMiniLM:**

- jina-reranker-v2 is **~15x faster** than bge-reranker-v2-m3 while being half the size — superior throughput for reranking 150 candidates per query.
- Flash Attention 2 support means long legal passages (up to 1024 tokens) are handled efficiently.
- Explicitly trained on multilingual query-document pairs including German. The existing mMiniLM was trained primarily on MS-MARCO passages, not legal German.
- 278M params fits comfortably within a single T4 GPU alongside other models in a sequential loading strategy.

**Why not bge-reranker-v2-m3:**

- Despite strong performance, it is ~560M params — twice the size of jina-reranker-v2 — and slower. For a 12-hour budget with 40 test queries × 150 candidates each, speed matters.

**Why not zerank-1 or zerank-1-small:**

- zerank-1 is API-only (no offline inference); zerank-1-small is open-source but 1.7B params — too large for the VRAM budget when running alongside BGE-M3 and Mistral-7B.

**Fallback:** Keep `mmarco-mMiniLMv2-L12-H384-v1` as the fast fallback (already in the project) if jina-reranker-v2 VRAM pressure causes OOM.

**Confidence:** MEDIUM-HIGH (official Jina benchmarks; legal-specific benchmarks not available)

---

### 4. Small LLM for Query Analysis (7B class)

| Recommendation | Model | Disk | VRAM (4-bit NF4) |
|----------------|-------|------|------------------|
| PRIMARY | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB fp16 / ~4 GB 4-bit | ~4–5 GB |
| ALTERNATIVE | `mistralai/Mistral-7B-Instruct-v0.3` | ~14 GB fp16 / ~4 GB 4-bit | ~3.5–4 GB |

**Why Qwen2.5-7B-Instruct over Mistral-7B:**

- Qwen2.5-7B significantly outperforms Mistral-7B-v0.3 on instruction following, multilingual text understanding, and structured output generation — all critical for query analysis and German legal language expansion.
- Qwen2.5-7B natively handles German text better than Mistral-7B due to broader multilingual training corpus.
- T4 compatibility confirmed: Qwen2.5-7B-Instruct with 4-bit bitsandbytes quantization runs at ~3.8 tok/s on T4, adequate for batch query analysis (40 test queries × ~100 tokens output = ~4000 tokens total, under 20 minutes).
- The `unsloth/Qwen2.5-7B-bnb-4bit` pre-quantized model is directly available on HuggingFace and loads without additional quantization step, saving ~5–10 minutes of Kaggle runtime.

**Why Mistral-7B remains the fallback:**

- Slightly lower VRAM footprint (~3.4 GB vs ~4–5 GB for Qwen2.5-7B 4-bit), which matters if you're tight on VRAM during concurrent model phases.
- More inference benchmarks are available for Mistral-7B on T4, reducing deployment risk.

**4-bit quantization setup:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Critical constraint:** Load the LLM only during query analysis phase. Unload (`del model; torch.cuda.empty_cache()`) before loading embedding model or reranker. Never keep all three in VRAM simultaneously.

**Confidence:** MEDIUM (T4 inference speed confirmed for Qwen2.5; Mistral-7B 4-bit VRAM confirmed; direct legal query analysis comparison not benchmarked)

---

### 5. Vector Search (FAISS)

| Recommendation | Package | Notes |
|----------------|---------|-------|
| PRIMARY | `faiss-cpu` (current) + GPU index via `faiss-gpu-cu12` | See below |
| Index type | `IndexIVFFlat` (laws) / `IndexFlatL2` (laws, if fits) | |

**Keep faiss-cpu as the foundation; add GPU index selectively:**

- The laws corpus has 175K documents × 1024 dimensions. An `IndexFlatL2` (exact search) for this corpus fits in ~730 MB of VRAM — trivially within T4's 16 GB. GPU exact search via `GpuIndexFlat` is 10–50x faster than CPU and is the right choice here.
- The court corpus (2.47M docs) cannot be dense-indexed within the 12-hour budget at encoding time (estimated 2.47M × 512 tokens at ~500 docs/sec on T4 = ~83 minutes encoding + ~10 GB+ VRAM for the index). BM25 + sampled dense rerank remains the correct strategy for courts.

**faiss-gpu installation on Kaggle (critical note):**

- `pip install faiss-gpu` is deprecated as of 1.7.3. Use:
  ```bash
  pip install faiss-gpu-cu12
  ```
  This is the CUDA 12 compatible PyPI wheel for Kaggle T4 (which runs CUDA 12.x).
- Alternatively, the pre-uploaded faiss-gpu Kaggle dataset (`tkm2261/faissgpu`) is a community-maintained offline installation option.

**Index strategy:**

```python
import faiss
# Laws corpus: exact GPU search
res = faiss.StandardGpuResources()
flat_index = faiss.IndexFlatIP(1024)  # inner product = cosine if normalized
gpu_index = faiss.index_cpu_to_gpu(res, 0, flat_index)
gpu_index.add(laws_embeddings)  # 175K × 1024 float32 = ~730 MB VRAM

# Query
D, I = gpu_index.search(query_embedding, k=100)
```

**Why not Qdrant/Milvus/Weaviate:**

- All require persistent server processes — impossible in a Kaggle offline notebook. FAISS is an in-process library with no server dependency. This is a hard constraint.

**Confidence:** HIGH (FAISS GPU behavior well-documented; Kaggle pip wheel confirmed via community)

---

### 6. Translation Model (en→de)

| Recommendation | Model | Size | VRAM |
|----------------|-------|------|------|
| PRIMARY | `Helsinki-NLP/opus-mt-en-de` | ~300 MB | ~500 MB CPU |
| UPGRADE OPTION | `Helsinki-NLP/opus-mt-tc-big-en-de` | ~1.2 GB | ~2 GB GPU |

**Why keep opus-mt-en-de (and upgrade to the big variant):**

- opus-mt-en-de is already working in the pipeline. For short query strings (10–50 words), its translation quality for en→de is adequate and competitive with NLLB-200 on high-resource language pairs like English-German.
- NLLB-200 (CC-BY-NC license) is excluded — the NC restriction creates IP risk for a competition submission.
- MADLAD-400 (10.7B params) is far too large for the VRAM budget.
- `opus-mt-tc-big-en-de` is the improved "big" Transformer variant trained on the same OPUS data. It produces noticeably better German output for domain-specific content and is still small (~1.2 GB). Run on CPU since it doesn't need GPU acceleration for 40–50 queries.

**Why translation matters less than it used to:**

- BGE-M3 handles cross-lingual retrieval natively (English query → German document) without query translation. Translation is primarily useful for BM25 keyword matching, not dense retrieval.
- Prioritize: use BGE-M3 dense without translation first; add German query translation only for BM25 sparse retrieval over the court corpus.

**Confidence:** MEDIUM (opus-mt-en-de quality confirmed in use; tc-big variant is LOW confidence for legal domain specifically — no legal benchmark data found)

---

## Complete Stack Summary

| Component | Library/Model | Version | Role |
|-----------|---------------|---------|------|
| Dense embedding | `BAAI/bge-m3` | latest | Cross-lingual retrieval, laws + court candidates |
| Sparse retrieval | `bm25s` | 0.2.x | BM25 over 2.47M court docs |
| Cross-encoder | `jinaai/jina-reranker-v2-base-multilingual` | latest | Reranking top candidates |
| LLM (query analysis) | `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) | Qwen2.5 | Query expansion, citation count estimation |
| Vector search | `faiss-gpu-cu12` | 1.7.4+ | GPU ANN for laws corpus |
| Translation | `Helsinki-NLP/opus-mt-tc-big-en-de` | latest | BM25 query translation |
| Deep learning | `torch` 2.1.0+ | 2.1+ | Model inference |
| Transformers | `transformers` 4.38.0+ | 4.38+ | Model loading |
| Quantization | `bitsandbytes` 0.41.0+ | 0.41+ | 4-bit LLM quantization |
| Embeddings lib | `FlagEmbedding` | 1.2.x | BGE-M3 specific API |
| Data processing | `pandas` 2.0+, `numpy` 1.24+  | — | CSV handling, scoring |

## Alternatives Considered and Rejected

| Category | Recommended | Rejected | Reason for Rejection |
|----------|-------------|----------|----------------------|
| Embedding | BGE-M3 | Qwen3-Embedding-8B | 8B params exceeds VRAM budget with other models loaded |
| Embedding | BGE-M3 | multilingual-e5-large | Lower cross-lingual performance on MIRACL; 512 token limit |
| BM25 | bm25s | rank-bm25 | 500x slower; prohibitive at 2.47M docs |
| BM25 | bm25s | Elasticsearch | Requires Java server; not compatible with offline Kaggle |
| Reranker | jina-reranker-v2 | bge-reranker-v2-m3 | 2x larger, slower throughput, same task |
| Reranker | jina-reranker-v2 | zerank-1 | API-only; not available offline |
| LLM | Qwen2.5-7B | Qwen3-8B | Qwen3-8B untested on T4; Qwen2.5-7B has confirmed T4 benchmarks |
| LLM | Qwen2.5-7B | Llama-3.1-8B | Qwen2.5-7B stronger German/multilingual performance |
| Translation | opus-mt-tc-big-en-de | NLLB-200 | CC-BY-NC license; legal risk for competition |
| Translation | opus-mt-tc-big-en-de | MADLAD-400 10.7B | Too large for VRAM budget |
| Vector DB | FAISS | Qdrant/Milvus/Weaviate | All require persistent server; incompatible with Kaggle notebooks |

## Installation

```bash
# Sparse retrieval (replaces rank-bm25 for court corpus)
pip install bm25s[full]

# BGE-M3 embedding (replaces multilingual-e5-large)
pip install FlagEmbedding

# FAISS GPU (replaces faiss-cpu for laws index)
pip install faiss-gpu-cu12

# 4-bit quantization for LLM
pip install bitsandbytes>=0.41.0 accelerate

# Upgraded translation model
# (no additional install — use existing transformers MarianMT)
```

## VRAM Allocation Map (T4 x2, 16 GB each)

| Phase | GPU 0 Load | GPU 1 Load | Notes |
|-------|-----------|-----------|-------|
| Encoding (laws) | BGE-M3 (~5 GB) + FAISS GPU index (~1 GB) | idle | Sequential; encode all laws first |
| BM25 court retrieval | CPU only (RAM ~12 GB for bm25s index) | idle | bm25s is CPU-only |
| Dense rerank (court top-K) | BGE-M3 (~5 GB) | idle | Encode court candidates only |
| Reranking | Unload BGE-M3; load jina-reranker-v2 (~2.5 GB) | idle | Sequential load |
| LLM query analysis | Unload reranker; load Qwen2.5-7B 4-bit (~4.5 GB) | available | LLM can use device_map="auto" across both GPUs |
| Translation | CPU (opus-mt ~500 MB RAM) | idle | Never needs GPU |

**Rule:** Never load more than one neural model simultaneously unless combining BGE-M3 + FAISS GPU index (which is safe at ~6 GB total).

---

## Sources

- MMTEB / MIRACL benchmark results: [https://arxiv.org/abs/2502.13595](https://arxiv.org/abs/2502.13595)
- BGE-M3 model card: [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- bm25s paper: [https://arxiv.org/abs/2407.03618](https://arxiv.org/abs/2407.03618)
- bm25s PyPI / 2M-doc benchmark: [https://bm25s.github.io/](https://bm25s.github.io/)
- Jina Reranker v2 HF model card: [https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
- Qwen2.5-7B T4 benchmark: [https://medium.com/@wltsankalpa/benchmarking-qwen-models-across-nvidia-gpus-t4-l4-h100-architectures-finding-your-sweet-spot-a59a0adf9043](https://medium.com/@wltsankalpa/benchmarking-qwen-models-across-nvidia-gpus-t4-l4-h100-architectures-finding-your-sweet-spot-a59a0adf9043)
- FAISS GPU cuVS: [https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- Mistral-7B 4-bit VRAM: [https://kaitchup.substack.com/p/mistral-7b-recipes-for-fine-tuning](https://kaitchup.substack.com/p/mistral-7b-recipes-for-fine-tuning)
- Zeroentropy reranker guide: [https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
