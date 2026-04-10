# Architecture Patterns: Multi-Stage Legal Retrieval Pipeline

**Domain:** Kaggle competition — Swiss legal citation retrieval
**Researched:** 2026-04-10
**Overall confidence:** HIGH (component design), MEDIUM (time estimates), LOW (exact VRAM numbers at runtime)

---

## Recommended Architecture

A **six-stage sequential pipeline** with resource-aware model lifecycle management. The two corpora are handled by parallel sub-pipelines that converge at fusion. LLM augmentation runs only on the 50 queries (val + test), making per-query cost tolerable.

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 0: Data Loading                                              │
│  laws_de.csv (175K) + court_considerations.csv (2.47M) + queries   │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│  STAGE 1: Index Building (CPU-only stage)                           │
│  ┌─────────────────────┐   ┌────────────────────────────────────┐   │
│  │  Laws BM25 Index    │   │  Court BM25 Index                  │   │
│  │  BM25S, 175K docs   │   │  BM25S, 2.47M docs                │   │
│  │  ~5 min             │   │  ~30-45 min                        │   │
│  └─────────────────────┘   └────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Laws FAISS Dense Index                                     │    │
│  │  multilingual-e5-large, 175K × 1024-dim, GPU               │    │
│  │  ~25-35 min                                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Entity Lookup Table (in-memory dict)                       │    │
│  │  Art.N + law_code → [doc_ids]                              │    │
│  │  ~5 min                                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Models: e5-large loaded then KEPT for query use
┌─────────────────▼───────────────────────────────────────────────────┐
│  STAGE 2: Per-Query Processing (runs for each of 50 queries)        │
│                                                                     │
│  2a. Query Analysis (LLM-optional path)                            │
│      ├─ Regex: extract Art.N + law_code entities                   │
│      ├─ Translation: OpusMT EN→DE (MarianMT, ~100MB VRAM)         │
│      └─ [LLM path] Mistral-7B 4-bit: German query expansion        │
│                                                                     │
│  2b. Entity-Driven Lookup (zero retrieval cost)                    │
│      ├─ Direct lookup: entities → doc_ids from lookup table        │
│      ├─ Nearby article expansion: Art.N±3 in same law             │
│      └─ Citation graph: law_code+Art.N → court decisions citing it │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│  STAGE 3: Retrieval (per query, two corpus sub-pipelines)           │
│                                                                     │
│  Laws Sub-pipeline:                                                 │
│  ├─ BM25 laws: top-100                                             │
│  └─ Dense laws (FAISS): top-100 (English query, already indexed)   │
│                                                                     │
│  Court Sub-pipeline:                                                │
│  ├─ BM25 court: top-500 candidates                                 │
│  └─ Dense court: encode top-500 BM25 docs on-the-fly → rerank top-200 │
│     (not pre-indexed — infeasible for 2.47M; sample-then-encode)   │
│                                                                     │
│  Entity-lookup results enter as a fourth ranked list               │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│  STAGE 4: Fusion + Reranking                                        │
│  ├─ RRF: fuse 4 lists (BM25-laws, dense-laws, BM25-court, entity) │
│  ├─ Take top-200 fused candidates                                  │
│  └─ Cross-encoder (mMiniLM): rerank top-200 → top-80 scored       │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│  STAGE 5: LLM Augmentation (Mistral-7B 4-bit)                      │
│  ├─ Load Mistral (unload e5-large + cross-encoder first)           │
│  ├─ Per-query: "Given this legal question, how many citations       │
│  │   are expected?" → estimated K                                  │
│  ├─ Optional: generate candidate citation strings for graph lookup  │
│  └─ Unload Mistral; reload cross-encoder for final pass if needed  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│  STAGE 6: Calibration + Output                                      │
│  ├─ Validation: grid-search top-K (1-80) maximizing macro-F1       │
│  ├─ LLM K estimate blended with calibrated K (weighted average)    │
│  └─ Write submission.csv                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Boundaries

| Component | Responsibility | Inputs | Outputs | State |
|-----------|---------------|--------|---------|-------|
| DataLoader | CSV parsing, text normalisation | CSV files | pandas DataFrames | stateless |
| BM25IndexBuilder | BM25S index for each corpus | DataFrame text column | BM25S objects | immutable after build |
| FAISSIndexBuilder | Embed laws + build IndexFlatIP | laws DataFrame, e5-large model | FAISS index + doc_ids array | immutable after build |
| EntityLookupBuilder | Build Art.N → [doc_id] dict; citation graph dict | laws DataFrame + court DataFrame | two dicts | immutable after build |
| QueryAnalyser | Entity extraction + translation + optional LLM expansion | raw query string | translated_query, entity_list, expanded_terms | stateless per query |
| LexicalRetriever | BM25S.retrieve() for both corpora | query tokens, BM25S objects | ranked [(doc_id, score)] per corpus | stateless |
| DenseRetriever | FAISS search for laws; on-the-fly encode+cosine for court candidates | query embedding, FAISS index, top-N doc texts | ranked [(doc_id, score)] | stateless |
| EntityRetriever | Direct lookup + graph expansion | entity_list, lookup dicts | [(doc_id, score=1.0)] with boost | stateless |
| RRFFuser | Reciprocal Rank Fusion of N ranked lists | list of ranked lists | single merged ranked list | stateless |
| CrossEncoderReranker | Pairwise query-doc scoring | query, candidate doc texts, mMiniLM model | [(doc_id, score)] sorted | stateless |
| MistralAugmentor | Query analysis, K estimation, citation generation | query string, Mistral model | estimated_K, optional expansions | stateless |
| Calibrator | Grid-search K on val set | val predictions, val gold | best_K, macro-F1 score | writes one scalar |
| OutputFormatter | Format predictions to submission CSV | test predictions, best_K | submission.csv | stateless |

**Communication rules:**
- Components pass plain Python objects (lists, dicts, DataFrames). No shared mutable state except the BM25/FAISS index objects which are read-only after build.
- Models are passed as constructor arguments, not globals. This makes sequential load/unload explicit.
- Entity lookup tables are built once from DataFrames before the query loop starts.

---

## Data Flow

```
CSV files
    │
    ▼
DataFrames (laws_df, court_df, val_df, test_df)
    │
    ├──► BM25S index (laws)  ─────────────────────────────┐
    ├──► BM25S index (court) ─────────────────────────────┤
    ├──► FAISS index (laws)  ─────────────────────────────┤
    └──► entity_lookup dict  ─────────────────────────────┤
                                                          │
    ┌─────────────────────────────────────────────────────┘
    │              (indices loaded, query loop begins)
    │
    ▼  per query:
raw_query
    │
    ├──► extract_entities()  ──────────────────────────────► entity_list
    ├──► translate()         ──────────────────────────────► german_query
    └──► [optional] mistral_expand() ─────────────────────► expanded_terms
         (only if Mistral loaded in Stage 5)
    │
    ▼
german_query + entity_list + expanded_terms
    │
    ├──► bm25_laws.retrieve(german_query)      → laws_ranked_100
    ├──► bm25_court.retrieve(german_query)     → court_ranked_500
    ├──► faiss_laws.search(english_embedding)  → dense_laws_100
    └──► entity_lookup.get(entity_list)        → entity_docs
         + graph_expand(entity_list)           → graph_docs
    │
    ▼
rrf_fuse([laws_ranked_100, court_ranked_500_reranked, dense_laws_100, entity_docs])
    → merged_200
    │
    ▼
cross_encoder.rerank(query, merged_200)
    → scored_80
    │
    ▼
[on val set] calibrate_K(scored_80, gold)
[on test set] top_K(scored_80, best_K or mistral_K)
    │
    ▼
submission.csv
```

**Key asymmetry:** Court corpus never gets a pre-built dense index. The 2.47M documents would take ~8+ hours to embed on T4. Instead: BM25 court top-500 → embed those 500 texts on-the-fly → cosine rank → keep top-200. This adds ~15-30s per query for the on-the-fly embedding step across 50 queries, which is acceptable.

---

## VRAM Budget

T4 x2 = 32GB total VRAM. Strategy: use GPU-0 for the active neural model, GPU-1 as overflow/spillover with `device_map="balanced_low_0"` for Mistral.

| Stage | Active Models | VRAM (GPU-0) | VRAM (GPU-1) | Total |
|-------|--------------|--------------|--------------|-------|
| Index build (Stage 1) | e5-large only | ~2.5GB | 0 | ~2.5GB |
| Query loop (Stages 2-4) | e5-large + translator + cross-encoder | ~2.5 + 0.3 + 0.9 = ~3.7GB | 0 | ~3.7GB |
| LLM stage (Stage 5) | Mistral-7B 4-bit only | ~4.5GB | ~0.5GB spillover | ~5GB |
| Final output | nothing | 0 | 0 | 0 |

**Confidence:** MEDIUM. Mistral-7B 4-bit is ~3.5-5GB base VRAM with bitsandbytes NF4; KV-cache for 50-query batch adds 0.5-1GB. multilingual-e5-large (560M params) is ~2.1GB in FP16. mMiniLM cross-encoder is ~450MB. These fit comfortably without unloading between Stages 2-4.

**Critical rule:** Unload e5-large and cross-encoder before loading Mistral. Reload cross-encoder after Mistral if a second reranking pass is wanted. Use `del model; torch.cuda.empty_cache(); gc.collect()` explicitly.

**Quantization for Mistral:**
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

---

## Time Budget

Total budget: 12 hours = 720 minutes. 40 test queries + 10 val queries = 50 queries total.

| Stage | Operation | Estimated Time | Confidence |
|-------|-----------|---------------|------------|
| Stage 0 | Load 2.47M court CSV into RAM | 8-15 min | MEDIUM |
| Stage 1a | Build BM25S laws index (175K docs) | 3-6 min | MEDIUM |
| Stage 1b | Build BM25S court index (2.47M docs) | 25-45 min | LOW |
| Stage 1c | Embed laws + build FAISS (175K × e5-large, GPU) | 20-35 min | MEDIUM |
| Stage 1d | Build entity lookup dict | 2-4 min | HIGH |
| Stage 2-4 | Per-query retrieval + reranking (50 queries) | 30-60 min | MEDIUM |
| Stage 4b | On-the-fly court dense re-score (50 × 500 docs) | 15-30 min | LOW |
| Stage 5 | Mistral load + 50-query augmentation | 20-40 min | LOW |
| Stage 6 | Calibration + output | 1-2 min | HIGH |
| **Total** | | **~2.5 to 4 hours** | MEDIUM |

**Budget margin:** Even pessimistic estimate leaves 8+ hours of headroom. The primary risk is BM25 court index RAM (2.47M docs × vocab size sparse matrix). If it OOMs at 30GB Kaggle RAM, fall back to chunked BM25 with top-1000 prefilter using TF-IDF (sklearn, lighter memory).

**BM25S vs rank-bm25 decision:** Use BM25S (not rank-bm25) for court corpus. rank-bm25 fails or becomes impractically slow beyond ~1M documents on 30GB RAM. BM25S sparse matrix approach is the only viable option for 2.47M docs. Laws corpus can still use rank-bm25 (175K is fine) but unify to BM25S for consistency.

---

## Build Order (Incremental, Each Layer Submittable)

Dependencies flow strictly top-to-bottom. Each layer below can be submitted independently.

```
Layer 0 (submit now): Fix GPU in Kaggle notebook
  └─ Prerequisite for all neural stages; baseline score improvement

Layer 1: Laws pipeline only, GPU-enabled
  ├─ BM25 laws + FAISS laws (e5-large) + cross-encoder rerank
  ├─ Calibrated K on val set
  └─ Submittable; establishes laws-only F1 baseline

Layer 2: Add court BM25 (no dense)
  ├─ BM25S court index + RRF fusion with laws results
  ├─ Depends on: Layer 1 (fusion logic already present)
  └─ Submittable; expect F1 jump because 41% val citations are court decisions

Layer 3: Add entity-driven retrieval
  ├─ Entity extraction (regex, already exists) + lookup table
  ├─ Citation graph expansion (court docs citing explicit Art.N)
  ├─ Nearby article expansion
  ├─ Depends on: Layer 2 (fourth RRF input added)
  └─ Submittable; highest value for queries with explicit article references

Layer 4: Add on-the-fly court dense reranking
  ├─ Take BM25 court top-500 → embed → cosine sort
  ├─ Depends on: Layer 2 (court BM25 candidates are input)
  └─ Submittable; adds semantic signal for court retrieval

Layer 5: Add Mistral-7B augmentation
  ├─ Query analysis + K estimation + German query expansion
  ├─ Depends on: Layer 1+ (needs working pipeline to augment)
  ├─ Load/unload management required (see VRAM section)
  └─ Submittable; highest implementation risk — build last
```

**Why this order:** Laws-only first because 59% of val citations are law articles, so Layer 1 alone should move F1 significantly. Court BM25 second because it addresses the most critical gap (41% citations missing entirely). Entity retrieval third because it requires no new models and is fast to build. Dense court reranking fourth because it adds quality to already-retrieved court candidates. Mistral last because it has highest complexity, is optional, and all prior layers run without it.

---

## Two-Corpus Retrieval Strategy

### Laws corpus (175K docs)
- **Full dense index:** Feasible. 175K × 1024-dim float32 = 715MB FAISS index, ~25-35 min to build on T4.
- **Strategy:** BM25S + FAISS dense, both pre-built at startup, queried per-query.

### Court corpus (2.47M docs)
- **Full dense index:** NOT feasible. Would require ~10GB storage + 8+ hours embedding on T4. Ruled out.
- **Strategy:** BM25S index (sparse, ~3-8GB RAM estimated) → top-500 candidates → embed those 500 texts with e5-large on GPU → cosine similarity → take top-200.
- **Why 500 BM25 candidates for dense reranking, not 100:** Legal court decisions can be verbose and lexically diverse. 500 improves recall before semantic reranking. Embedding 500 texts on GPU takes ~3-5 seconds, acceptable across 50 queries.

### Entity retrieval (both corpora)
- **Pre-built lookup dict:** `{(law_code, art_num): [law_doc_id, ...]}` — built once from laws_df text parsing.
- **Citation graph dict:** `{(law_code, art_num): [court_doc_id, ...]}` — built by scanning court consideration text for "Art. N law_code" mentions. This is the most expensive lookup to build (~5-10 min for 2.47M docs) but eliminates per-query graph traversal.
- **Nearby expansion:** At query time only — dict lookup for art_num±3 requires no graph structure.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Pre-build dense index for court corpus
**What:** Attempting to embed all 2.47M court documents into FAISS at startup.
**Why bad:** 2.47M × 1024-dim × 4 bytes = ~10GB just for the embeddings. Building takes 8+ hours on T4. Exceeds both VRAM and time budget.
**Instead:** BM25 first-pass → on-the-fly dense reranking of top-500 BM25 hits.

### Anti-Pattern 2: Keep all models in VRAM simultaneously
**What:** Loading e5-large + cross-encoder + Mistral concurrently.
**Why bad:** e5-large (~2.1GB) + cross-encoder (~0.5GB) + Mistral-7B 4-bit (~5GB) = ~7.6GB which fits on one T4, BUT KV cache + activation memory during inference pushes peak well above 16GB.
**Instead:** Sequential lifecycle: e5-large loaded for index build + query embedding → cross-encoder loaded for reranking → unload both → load Mistral → unload Mistral.

### Anti-Pattern 3: Using rank-bm25 for court corpus
**What:** Building a rank-bm25 BM25Okapi object on 2.47M documents.
**Why bad:** rank-bm25 stores all token lists in Python lists. At 2.47M documents this will OOM on 30GB Kaggle RAM or take 45+ minutes.
**Instead:** BM25S with scipy sparse matrix storage. Same BM25 formula, 500x faster queries, ~50% lower memory than rank-bm25.

### Anti-Pattern 4: Running Mistral for every query before pipeline is stable
**What:** Making LLM augmentation a required stage during early development.
**Why bad:** If Mistral fails to load or times out, the entire submission fails. Every submission costs 12 hours.
**Instead:** Mistral stage is always optional and gated by a `USE_LLM_AUGMENTATION = True/False` flag. The pipeline must produce valid output without it.

### Anti-Pattern 5: Calibrating on test set
**What:** Using test query predictions to choose top-K threshold.
**Why bad:** No ground truth for test set. Calibration must use validation set only, then apply the learned K to test.
**Instead:** Current calibration logic in solution.py is correct — preserve it exactly.

---

## Scalability Considerations

This system is not designed to scale beyond Kaggle competition constraints. Key one-time design choices:

| Concern | Current approach | Limit reached when |
|---------|-----------------|-------------------|
| BM25 court index RAM | BM25S sparse matrix, ~30GB cap | >5M docs or <30GB RAM |
| FAISS laws index | IndexFlatIP in-memory, ~715MB | >500K docs at 1024-dim |
| Mistral VRAM | 4-bit NF4, single T4 | Context length >4K tokens |
| Query throughput | Sequential per query | Acceptable for 50 queries; no parallelism needed |

If competition adds more queries, the 50-query assumption becomes invalid and per-query LLM cost needs re-evaluation.

---

## Sources

- BM25S paper and benchmarks: https://arxiv.org/html/2407.03618v1
- BM25S HuggingFace blog (memory-mapping, scipy sparse): https://huggingface.co/blog/xhluca/bm25s
- Mistral-7B 4-bit VRAM usage (~3.5-5GB): https://kaitchup.substack.com/p/mistral-7b-recipes-for-fine-tuning
- Multi-phase legal retrieval architecture: https://arxiv.org/html/2403.18093
- Kaggle T4x2 specs (16GB each, 30GB RAM): https://www.kaggle.com/discussions/product-feedback/361328
- multilingual-e5-large on T4: https://github.com/inferless/multilingual-e5-large
- Multi-GPU model sharding with device_map: https://huggingface.co/docs/diffusers/training/distributed_inference

---

*Architecture analysis: 2026-04-10*
