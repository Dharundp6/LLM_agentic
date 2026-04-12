# Roadmap: LLM Agentic Legal Information Retrieval

## Overview

Starting from a near-zero F1 baseline (0.009), this roadmap builds the Kaggle submission in five independently-submittable layers. Each phase fixes a distinct root cause of the score deficit: silent CPU fallback and missing court corpus are the dominant failures; entity-driven precision signals and dense reranking are multipliers; LLM augmentation is the highest-risk, highest-upside capstone. The pipeline is refactored from the existing solution.py into a GPU-enabled Kaggle notebook incrementally, with every phase producing a valid submission.csv.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Foundation + Laws Pipeline** - GPU-enabled laws retrieval (BGE-M3 + BM25) with calibrated output; baseline submission
- [ ] **Phase 2: Court Corpus Coverage** - BM25 over all 2.47M court decisions; RRF fusion; expected F1 jump to 0.08-0.15
- [ ] **Phase 3: Entity + Citation Graph Signals** - Regex entity extraction, direct statute lookup, citation graph expansion; highest precision gains
- [ ] **Phase 4: Quality Signals** - Jina cross-encoder reranking on full fused candidate pool; BGE-M3 dense reranking of court BM25 top-500
- [ ] **Phase 5: LLM Augmentation** - Qwen2.5-7B 4-bit HyDE query expansion and per-query citation count estimation; optional, gated flag

## Phase Details

### Phase 1: Foundation + Laws Pipeline
**Goal**: Notebook runs on GPU with BGE-M3 and produces a calibrated laws-only submission that is structurally correct and independently submittable
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, LAWS-01, LAWS-02, LAWS-03, LAWS-04, LAWS-05, QUERY-03, QUERY-04, CALIB-01, CALIB-02, CALIB-03, CALIB-04, CALIB-05, INC-01, INC-02, INC-03
**Success Criteria** (what must be TRUE):
  1. Notebook first cell asserts `torch.cuda.is_available()` and logs device name — silent CPU fallback is impossible
  2. Full pipeline completes end-to-end in under 11 hours (>=1 hour safety margin) with smoke test on 3 val queries finishing in <5 minutes
  3. BGE-M3 (bge-m3, 1024-dim) is loaded on CUDA; embedding dimension assert passes; only one model on GPU at a time with explicit `cuda.empty_cache()` between stages
  4. Val Macro-F1 is logged after each pipeline change; `submission.csv` passes format validation (all test IDs present, no NaN, semicolons not commas, no trailing semicolons)
  5. Per-query prediction count targets ~20-30 citations (not fixed 100); calibration uses val set only and never touches test
**Plans**: TBD

### Phase 2: Court Corpus Coverage
**Goal**: BM25 retrieval over all 2.47M court decisions is added to the pipeline; RRF fuses laws and court signals; court recall is non-zero for the first time
**Depends on**: Phase 1
**Requirements**: COURT-01, COURT-02, COURT-03, COURT-05, QUERY-01, FUSE-01, FUSE-05
**Success Criteria** (what must be TRUE):
  1. bm25s indexes German-only court_considerations.csv (filtered, not full 2.47M); index build completes within Kaggle 30GB RAM budget (COURT-05 profiled)
  2. BM25 court retrieves top-500 candidates per query; predicted citations include BGE/BGer docket references (court recall > 0%)
  3. Weighted RRF fuses laws-dense, laws-BM25, and court-BM25 signals with documented per-signal weights and k=60 default; adding court BM25 does not decrease val Macro-F1
  4. EN->DE translation via opus-mt-tc-big-en-de feeds German query tokens to court BM25; each retrieval signal is evaluated independently on val before combining
  5. Submission.csv produced with court citations included; expected Val Macro-F1 >= 0.05 (minimum viability for court coverage)
**Plans:** 2 plans
Plans:
- [ ] 02-01-PLAN.md -- Court corpus load, German filter, bm25s index build with RAM profiling, BM25 court retrieval top-200, per-signal val F1 logging
- [ ] 02-02-PLAN.md -- 3-signal RRF fusion with COURT_ID_OFFSET namespace, cross-encoder + calibration + submission updated for dual-corpus

### Phase 3: Entity + Citation Graph Signals
**Goal**: Queries with explicit article references (Art. X OR, BGE X Y Z) are answered with perfect-precision direct lookup; citation graph converts laws hits into court hits at zero additional model cost
**Depends on**: Phase 2
**Requirements**: ENT-01, ENT-02, ENT-03, ENT-04, ENT-05, QUERY-02, FUSE-02
**Success Criteria** (what must be TRUE):
  1. Regex extracts `(law_code, article_number)` tuples from queries; canonical Swiss citation format variants (Art./Abs. spacing, Roman numerals, ss/ß, ZGB/CC aliases) are normalized before matching
  2. Entity direct lookup returns exact-match statute doc_ids as a separate high-weight RRF signal; queries with explicit article refs show higher precision than retrieval-only baseline
  3. Citation graph inverted index `(law_code, art_num) -> [court_doc_ids]` is built once at startup from a single scan of court_considerations.csv; graph expansion adds BGE decisions that cite matched articles
  4. RRF signal weights are learned from the validation set via grid search across the now four-signal list (laws-dense, laws-BM25, court-BM25, entity-lookup); learned weights replace equal-weight default
  5. Submission.csv produced; Val Macro-F1 improves over Phase 2 baseline; nearby article expansion (+/-2 articles, same law) is present as a low-weight candidate path
**Plans**: TBD

### Phase 4: Quality Signals
**Goal**: Semantic quality of court candidates is improved via BGE-M3 on-the-fly dense reranking; jina-reranker-v2 cross-encoder replaces mMiniLM and reranks the full fused candidate pool
**Depends on**: Phase 3
**Requirements**: COURT-04, FUSE-03, FUSE-04
**Success Criteria** (what must be TRUE):
  1. Top-500 BM25 court candidates per query are encoded on-the-fly with BGE-M3 and reranked by cosine similarity before entering RRF fusion; total per-query dense reranking latency across 50 queries fits within time budget
  2. jina-reranker-v2-base-multilingual cross-encoder reranks the top-150 merged candidates; inference is batched at 32 pairs per forward pass on GPU
  3. Cross-encoder candidate count capped at 150 (not 500); document text truncated to 256 tokens; reranking phase completes in <30 minutes for all 50 queries
  4. Val Macro-F1 improves over Phase 3 baseline; pipeline still produces valid submission.csv with LLM_AUGMENTATION flag disabled
**Plans**: TBD

### Phase 5: LLM Augmentation
**Goal**: Qwen2.5-7B 4-bit generates HyDE German legal passages for query expansion and estimates per-query citation count; both signals are blended with calibrated baseline; LLM stage remains optional and gated
**Depends on**: Phase 4
**Requirements**: LLM-01, LLM-02, LLM-03, LLM-04, LLM-05, LLM-06
**Success Criteria** (what must be TRUE):
  1. Qwen2.5-7B-Instruct loaded with NF4 4-bit quantization, `bnb_4bit_compute_dtype=torch.float16`, `attn_implementation="eager"`; no bfloat16, no Flash Attention 2 (T4 is compute capability 7.5)
  2. LLM is loaded only during the augmentation stage and fully unloaded (del + cuda.empty_cache) before reranking; VRAM does not exceed T4 budget at any stage
  3. HyDE expansion generates 2-3 hypothetical German legal passages per query; averaged embeddings are combined with direct-query embedding and fed to laws dense retrieval
  4. LLM-estimated per-query citation count blends with calibrated K; all LLM-generated citation strings are fuzzy-snapped to canonical corpus form (Levenshtein <=2) before inclusion -- raw LLM output never appears in submission
  5. `USE_LLM_AUGMENTATION = False` produces a valid submission.csv identical in structure to Phase 4 output; val F1 delta measured before committing to test submission with LLM enabled
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation + Laws Pipeline | 0/TBD | Not started | - |
| 2. Court Corpus Coverage | 0/2 | Planned | - |
| 3. Entity + Citation Graph Signals | 0/TBD | Not started | - |
| 4. Quality Signals | 0/TBD | Not started | - |
| 5. LLM Augmentation | 0/TBD | Not started | - |
