# Requirements: LLM Agentic Legal Information Retrieval

**Defined:** 2026-04-10
**Core Value:** Retrieve correct Swiss legal citations across language boundaries — every missed citation is a direct hit to F1 score.

## v1 Requirements

Requirements for the competitive Kaggle submission. Each maps to a roadmap phase. Success metric: **Val Macro-F1 ≥ 0.25** (competitive top-tier target per PROJECT.md).

### Foundation (Runtime + GPU)

- [ ] **FOUND-01**: Kaggle notebook asserts `torch.cuda.is_available()` as its first action — fail fast, not silently
- [ ] **FOUND-02**: All neural models (embeddings, reranker, LLM) are confirmed loaded on CUDA device with logged device name
- [ ] **FOUND-03**: Sequential model lifecycle — only one neural model on GPU at a time; explicit `torch.cuda.empty_cache()` between stages
- [ ] **FOUND-04**: Kaggle/local runtime detection with conditional paths for data, models, and output directories
- [ ] **FOUND-05**: End-to-end smoke test on 3 val queries completes in <5 minutes before full inference run
- [ ] **FOUND-06**: Full pipeline completes within the 12-hour Kaggle runtime limit with ≥1 hour safety margin

### Laws Corpus Retrieval

- [ ] **LAWS-01**: BGE-M3 dense embedding model replaces multilingual-e5-large for laws corpus (`BAAI/bge-m3`, fp16)
- [ ] **LAWS-02**: FAISS GPU index (IndexFlatIP) built over all 175K laws with L2-normalized BGE-M3 embeddings
- [ ] **LAWS-03**: bm25s sparse index over laws corpus with German decompounding and no stemming (per PITFALLS research)
- [ ] **LAWS-04**: Laws sub-pipeline returns dense top-K + BM25 top-K candidates per query
- [ ] **LAWS-05**: Laws corpus text is preprocessed with canonical Swiss citation format (normalized whitespace, Art./Abs. spacing, Roman/Arabic normalization)

### Court Corpus Retrieval

- [ ] **COURT-01**: bm25s (not rank-bm25) indexes all 2.47M rows of `court_considerations.csv`
- [ ] **COURT-02**: Court corpus filtered to German-only rows before indexing to reduce index size and noise
- [ ] **COURT-03**: BM25 retrieves top-500 court candidates per query within time budget
- [ ] **COURT-04**: BGE-M3 on-the-fly dense reranking of the top-500 BM25 court candidates (encoded at query time, not pre-indexed)
- [ ] **COURT-05**: BM25 court index build profiled end-to-end to confirm it fits in Kaggle 30GB RAM budget

### Query Processing

- [ ] **QUERY-01**: English queries translated to German via `Helsinki-NLP/opus-mt-tc-big-en-de` for BM25 keyword matching
- [ ] **QUERY-02**: Regex entity extraction parses explicit citations from queries (Art. X, Abs. Y, law codes OR/ZGB/StPO/SchKG/BGB, SR numbers, BGE references)
- [ ] **QUERY-03**: BGE-M3 receives both English and German query forms for dense retrieval (handles cross-lingual natively — no translation required for dense path)
- [ ] **QUERY-04**: Query canonicalization normalizes Swiss citation format variants before entity matching

### Entity & Graph Signals

- [ ] **ENT-01**: Direct statute lookup by extracted `(law_code, article_number)` tuples — exact-match against laws corpus
- [ ] **ENT-02**: Nearby-article expansion adds adjacent article numbers (±2) in the same law as low-weight candidates
- [ ] **ENT-03**: Pre-built citation graph: `(law_code, art_num) → [court_doc_ids]` inverted index scanned once over court_considerations text at startup
- [ ] **ENT-04**: Citation graph expansion — when an article is found in laws, automatically add BGE court decisions that cite it
- [ ] **ENT-05**: Entity lookup signal contributes to RRF fusion as a separate high-weight rank list

### Fusion & Reranking

- [ ] **FUSE-01**: Weighted RRF fuses laws-dense, laws-BM25, court-BM25, court-dense-rerank, and entity-lookup signal lists
- [ ] **FUSE-02**: RRF signal weights learned from validation set via grid search (not equal weights)
- [ ] **FUSE-03**: jina-reranker-v2-base-multilingual cross-encoder reranks the top-150 merged candidates
- [ ] **FUSE-04**: Cross-encoder input is batched (32 pairs per forward pass) for T4 throughput
- [ ] **FUSE-05**: RRF `k=60` default with documented tuning range for per-signal weights

### LLM Augmentation

- [ ] **LLM-01**: Qwen2.5-7B-Instruct loaded with 4-bit NF4 quantization, `bnb_4bit_compute_dtype=torch.float16`, `attn_implementation="eager"` (T4-safe)
- [ ] **LLM-02**: LLM loaded only for the augmentation stage; unloaded before reranking to free VRAM
- [ ] **LLM-03**: HyDE-style query expansion generates 2-3 hypothetical German legal passages per query; averaged embeddings combined with direct-query embedding
- [ ] **LLM-04**: LLM estimates per-query citation count; result caps/adjusts the global K calibration baseline
- [ ] **LLM-05**: LLM augmentation is gated by `USE_LLM_AUGMENTATION` flag — pipeline must produce valid submission.csv with flag disabled
- [ ] **LLM-06**: All LLM-generated citations are fuzzy-snapped to canonical corpus form (Levenshtein ≤2); raw LLM output never appears in final submission

### Output Calibration

- [ ] **CALIB-01**: Per-query citation count calibrated on val set; target distribution ~20-30 predictions per query (not fixed 100)
- [ ] **CALIB-02**: Final predictions canonicalized to exact Swiss citation format expected by scorer (spacing, Art./Abs. punctuation, OR/ZGB vs CC abbreviation language, ss vs ß)
- [ ] **CALIB-03**: `submission.csv` format validated against competition specification before writing
- [ ] **CALIB-04**: Val Macro-F1 computed locally after each pipeline change; logged to track incremental improvements
- [ ] **CALIB-05**: Calibration never touches test set — K is tuned on val only

### Incremental Submission

- [ ] **INC-01**: Each of the 5 roadmap phases produces an independently submittable, valid `submission.csv`
- [ ] **INC-02**: Pipeline feature flags allow disabling later-phase components for A/B comparison on val
- [ ] **INC-03**: Git tags mark each phase's submission-ready state for reproducibility

## v2 Requirements

Deferred after primary leaderboard entry — not in current roadmap.

### Post-Competition Enhancements

- **V2-01**: FR/IT court decision retrieval if initial German-only results are insufficient
- **V2-02**: Contrastive LoRA fine-tuning of BGE-M3 on train.csv query-citation pairs
- **V2-03**: Multi-hop citation graph (decisions cited by decisions cited by laws)
- **V2-04**: Alternative LLMs (Qwen2.5-14B, DeepSeek-V2-Lite) if VRAM headroom found

## Out of Scope

Explicitly excluded — documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Full dense encoding of 2.47M court corpus | ~8 hours encoding time exceeds 12-hour budget; BM25 + on-the-fly dense rerank is the correct pattern |
| Pre-uploaded FAISS index as Kaggle dataset | User chose Kaggle-only constraint; no external datasets allowed |
| Fine-tuning embedding models within Kaggle | Requires days not hours, even with LoRA; infeasible in runtime budget |
| FR/IT court decision retrieval (v1) | German-heavy val/test composition; effort not justified for first submission |
| LLM-generated citation lists (hallucination-based) | Hallucinated citations score 0; Qwen has no Swiss law training data |
| rank-bm25 for court corpus | OOMs / times out on 2.47M docs; bm25s is 500x faster and memory-mapped |
| Full cross-encoder rerank of court corpus | O(n) inference on 2.47M docs is infeasible on any hardware |
| Elastic/Solr/external search engines | No internet in Kaggle offline notebooks |
| Fine-tuned OpusMT on legal German | Training time exceeds runtime budget; pretrained version is good enough |
| Flash Attention 2 for Qwen | T4 compute capability 7.5 < required 8.0 — will crash at kernel load |
| bfloat16 compute on T4 | T4 doesn't support bfloat16 → silent corruption; must use float16 |
| Creative prize track entry | User goal is cash prize / leaderboard placement, not creative track |

## Traceability

Maps each v1 requirement to exactly one phase. Populated 2026-04-11.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Pending |
| FOUND-02 | Phase 1 | Pending |
| FOUND-03 | Phase 1 | Pending |
| FOUND-04 | Phase 1 | Pending |
| FOUND-05 | Phase 1 | Pending |
| FOUND-06 | Phase 1 | Pending |
| LAWS-01 | Phase 1 | Pending |
| LAWS-02 | Phase 1 | Pending |
| LAWS-03 | Phase 1 | Pending |
| LAWS-04 | Phase 1 | Pending |
| LAWS-05 | Phase 1 | Pending |
| QUERY-03 | Phase 1 | Pending |
| QUERY-04 | Phase 1 | Pending |
| CALIB-01 | Phase 1 | Pending |
| CALIB-02 | Phase 1 | Pending |
| CALIB-03 | Phase 1 | Pending |
| CALIB-04 | Phase 1 | Pending |
| CALIB-05 | Phase 1 | Pending |
| INC-01 | Phase 1 | Pending |
| INC-02 | Phase 1 | Pending |
| INC-03 | Phase 1 | Pending |
| COURT-01 | Phase 2 | Pending |
| COURT-02 | Phase 2 | Pending |
| COURT-03 | Phase 2 | Pending |
| COURT-05 | Phase 2 | Pending |
| QUERY-01 | Phase 1 | Pending |
| FUSE-01 | Phase 2 | Pending |
| FUSE-05 | Phase 2 | Pending |
| ENT-01 | Phase 3 | Pending |
| ENT-02 | Phase 3 | Pending |
| ENT-03 | Phase 3 | Pending |
| ENT-04 | Phase 3 | Pending |
| ENT-05 | Phase 3 | Pending |
| QUERY-02 | Phase 3 | Pending |
| FUSE-02 | Phase 3 | Pending |
| COURT-04 | Phase 4 | Pending |
| FUSE-03 | Phase 4 | Pending |
| FUSE-04 | Phase 4 | Pending |
| LLM-01 | Phase 5 | Pending |
| LLM-02 | Phase 5 | Pending |
| LLM-03 | Phase 5 | Pending |
| LLM-04 | Phase 5 | Pending |
| LLM-05 | Phase 5 | Pending |
| LLM-06 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 43 total
- Mapped to phases: 43 ✓ (each requirement appears exactly once)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-10*
*Last updated: 2026-04-11 — traceability filled by roadmapper; all 43 requirements mapped*
