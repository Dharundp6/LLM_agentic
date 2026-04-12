# Phase 2: Court Corpus Coverage - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 adds **court decision retrieval** to the existing laws-only pipeline from Phase 1.
The 2.47M-row `court_considerations.csv` corpus is indexed with bm25s, queried with the
German translations already cached in Phase 1 Cell 4, and fused via RRF with the existing
laws-dense and laws-BM25 signals. Court recall goes from 0% to non-zero for the first time.

**In scope:** German-only court corpus filtering (COURT-02), bm25s court index build with
RAM profiling (COURT-01, COURT-05), BM25 court retrieval top-200 per query (COURT-03),
3-signal weighted RRF fusion (FUSE-01), per-signal independent val F1 logging (FUSE-05),
submission.csv updated with court citations.

**Out of scope (later phases):** BGE-M3 dense court reranking (Phase 4, COURT-04),
jina-reranker-v2 (Phase 4, FUSE-03), entity-driven direct lookup + citation graph
(Phase 3), learned RRF weights (Phase 3, FUSE-02), LLM augmentation (Phase 5).

**Dependency on Phase 1:** Uses `translations`, `bm25_query_texts`, `canonicalize()`,
`tokenize_for_bm25_de()`, `extract_legal_codes()`, `rrf_fuse()`, `calibrate_top_k()`,
`validate_submission()`, and the `unload()` helper -- all defined in Cells 0-13.

</domain>

<decisions>
## Implementation Decisions

### Court Corpus Filtering (COURT-02)

- **D-08:** **Filter court_considerations.csv to German-only rows before indexing.**
  The query pipeline produces German-language BM25 tokens (via OpusMT translation).
  Non-German court texts (French, Italian) will not match and only inflate the index.
  Detection method: use a simple heuristic (e.g., check for common German stopwords
  like "der", "die", "das", "und" presence) -- exact method is Claude's Discretion.
  Log the filtered count and percentage.

### RAM Budget & Chunked Build (COURT-05)

- **D-09:** **Build bm25s court index in chunks with explicit memory profiling.**
  2.47M docs x tokenized lists could approach or exceed the Kaggle 30GB RAM budget.
  Strategy: (1) tokenize in chunks of 500K docs, logging RSS after each chunk,
  (2) build the bm25s index from the full tokenized list, (3) if RSS exceeds 25GB
  at any point, log a warning and consider reducing the corpus (e.g., filter by
  text length > 50 chars to remove stubs). This is a profiling cell, not a blind
  build.

### BM25 Court Retrieval K (COURT-03)

- **D-10:** **Retrieve top-200 court candidates per query (BM25_COURT_K=200).**
  Matches the `solution.py` default. 200 is wide enough for recall; downstream
  reranking (Phase 4) will trim. Log per-query retrieval time; warn if mean
  exceeds 2 seconds (potential scaling issue for 40 queries x 200 candidates).

### RRF Fusion Strategy (FUSE-01, FUSE-05)

- **D-11:** **3-signal equal-weight RRF with k=60 default.** Signals:
  (1) laws-dense (from Phase 1 Cell 7), (2) laws-BM25 (from Phase 1 Cell 8),
  (3) court-BM25 (new in Phase 2). Each signal is an independent ranked list
  per query. `rrf_fuse()` from Phase 1 already handles arbitrary list counts.
  Per-signal weights are equal for now; learned weights (FUSE-02) are Phase 3.

- **D-12:** **Log per-signal val F1 independently before fusion.** For each of the
  three signals, compute val Macro-F1 using only that signal's top-K predictions.
  This provides a baseline to measure fusion lift and diagnose which signals help.
  Print a comparison table:
  ```
  === Per-Signal Val F1 ===
  laws-dense:  F1=X.XXXX @ top-K
  laws-BM25:   F1=X.XXXX @ top-K
  court-BM25:  F1=X.XXXX @ top-K
  fused-3sig:  F1=X.XXXX @ top-K
  ```
  Phase 2 success criterion: `fused-3sig >= max(individual signals)`. If adding
  court BM25 decreases F1, investigate before proceeding.

### Translation Reuse (QUERY-01)

- **D-13:** **QUERY-01 is already satisfied by Phase 1 Cell 4.** The `translations`
  and `bm25_query_texts` dicts are cached in memory after Cell 4 executes. Phase 2
  cells simply read `bm25_query_texts[qid]` for court BM25 queries -- no new
  translation code needed. The ROADMAP references `opus-mt-tc-big-en-de` which does
  not exist on HuggingFace; Phase 1 already switched to `opus-mt-en-de` (standard
  Helsinki-NLP model). This is the production translation model going forward.

### Cross-Encoder Reranking

- **D-14:** **Phase 2 reuses the mmarco cross-encoder from Phase 1 (Cell 10) on the
  FUSED candidate pool.** The cross-encoder now reranks the merged 3-signal fused
  list instead of the 2-signal laws-only list. No model change; just a wider input
  pool. RERANK_K stays at 150 (top-150 fused candidates per query).

### Claude's Discretion

The planner/executor has flexibility on:

- **Notebook cell organization.** Default: add 1-2 new cells between Cell 8 (bm25s
  laws) and Cell 9 (RRF fuse) for court load + filter + index + retrieve. Then
  modify Cell 9 to accept 3 signal lists instead of 2. Alternatively, extend
  Cell 8 with a court section -- either approach is fine.
- **Court text preprocessing.** Default: apply `canonicalize()` to court text before
  tokenization (same as laws). If court texts have different formatting patterns
  (e.g., docket numbers like "1B_123/2020"), extend canonicalize() minimally.
- **Language detection method.** Default: simple heuristic (German stopword ratio)
  over `langdetect` to avoid adding a dependency. Either is acceptable.
- **BM25_COURT_K tuning.** Start at 200; if court recall is surprisingly low, try
  500 on val before escalating to Phase 4's dense reranking.
- **Feature flag.** Default: `USE_COURT_CORPUS = True` (flip the existing flag from
  Phase 1 Cell 0). The flag gates all court-related cells so the pipeline can be
  toggled back to laws-only for debugging.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level specs
- `.planning/PROJECT.md` -- overall goal, constraints
- `.planning/REQUIREMENTS.md` -- COURT-01..05, FUSE-01, FUSE-05
- `.planning/ROADMAP.md` Phase 2 -- goal, depends-on, success criteria

### Phase 1 artifacts (dependency)
- `.planning/phases/01-foundation-laws-pipeline/01-CONTEXT.md` -- D-01 through D-07
- `.planning/phases/01-foundation-laws-pipeline/01-RESEARCH.md` -- Court Corpus section
- `notebook_kaggle.ipynb` -- Cells 0-13 (full Phase 1 pipeline)
- `solution.py` -- lines 100-114 (BM25 build pattern), 220-226 (rrf_fuse)

### Data files
- `Data/court_considerations.csv` -- the 2.47M court corpus
- `Data/laws_de.csv` -- the 175K laws corpus (Phase 1, for reference)
- `Data/val.csv` -- validation queries (10 queries with gold citations)

### Research artifacts
- `.planning/research/PITFALLS.md` -- especially CP-1 (German compound blindness)
- `.planning/research/STACK.md` -- bm25s justification

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable from Phase 1
- **`tokenize_for_bm25_de(text)`** (Cell 3) -- German-aware tokenizer with CharSplit
  decompounding and NLTK stopword removal. Reuse directly for court corpus tokenization.
- **`canonicalize(s)`** (Cell 3) -- Swiss citation format normalizer. Apply to court texts.
- **`rrf_fuse(ranked_lists, k=60)`** -- already accepts arbitrary number of ranked lists.
  Just pass 3 lists instead of 2.
- **`bm25_query_texts[qid]`** (Cell 4) -- cached German query + legal codes. Court BM25
  uses this directly.
- **`calibrate_top_k(val_results)`** (Cell 11) -- reuse for 3-signal calibration.
- **`validate_submission(df, test)`** (Cell 12) -- reuse unchanged.
- **Feature flags** (Cell 0) -- `USE_COURT_CORPUS = False` already defined; flip to True.

### Integration Points
- **Between Cell 8 and Cell 9:** Insert court cells here. Cell 8 produces `bm25_laws_ids`;
  new court cells produce `bm25_court_ids`. Cell 9 RRF fuse receives both.
- **Cell 9 modification:** `rrf_fuse([dense_laws_ids[qid], bm25_laws_ids[qid]])` becomes
  `rrf_fuse([dense_laws_ids[qid], bm25_laws_ids[qid], bm25_court_ids[qid]])`.
- **Cell 11 modification:** `val_results` must include court citations in predicted set.

### Key Risk
- **RAM:** 2.47M tokenized docs in Python lists + bm25s internal structures could peak
  at 15-25GB. Kaggle T4 has 30GB system RAM. Profile before committing.

</code_context>

<specifics>
## Specific Ideas

- **User priority signal (from Phase 1):** "I just want to win the competition."
  Optimize for val Macro-F1 improvement. Phase 2 success is measured by the delta
  from Phase 1 laws-only F1.
- **Expected F1 range:** Phase 2 target is 0.05-0.15. Adding court decisions covers
  the 41% of val gold citations that were previously invisible. Even weak court BM25
  recall should produce a measurable F1 jump.
- **Notebook is running on Kaggle right now (Phase 1 v11).** Phase 2 planning can
  proceed in parallel. When Phase 1 results arrive, use the actual val F1 as the
  baseline for Phase 2 success measurement.

</specifics>

<deferred>
## Deferred Ideas

- **Dense court reranking (COURT-04):** Deferred to Phase 4. Would encode court BM25
  top-500 with BGE-M3 at query time -- too slow for Phase 2 scope.
- **Learned RRF weights (FUSE-02):** Deferred to Phase 3. Need the entity/citation
  graph signal present before learning weights across all signals.
- **Court corpus deduplication:** Not in scope. Some court docs may have near-duplicate
  considerations. Address only if it causes measurable F1 issues.
- **Pre-encoded court embeddings (Strategy C from report):** Deliberately replaced with
  BM25-only court retrieval for now. Revisit if court BM25 recall is insufficient.

*No cross-referenced todos found.*

</deferred>

---

*Phase: 02-court-corpus-coverage*
*Context gathered: 2026-04-12*
