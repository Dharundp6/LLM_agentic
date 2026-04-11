# Phase 1: Foundation + Laws Pipeline - Context

**Gathered:** 2026-04-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 delivers the **first independently submittable layer** of the Kaggle pipeline:
a GPU-enabled notebook that retrieves over the **laws corpus only** (175K rows of
`laws_de.csv`) using **BGE-M3 dense + bm25s sparse**, fuses with RRF, optionally
reranks with a cross-encoder, and produces a calibrated `submission.csv` that passes
format validation.

**In scope:** GPU enablement + hard assert, sequential model lifecycle, BGE-M3 fp16
laws encoding, FAISS IndexFlatIP over laws, bm25s laws index with German decompounding,
RRF fusion of the two laws signals, K calibration on val set, smoke test on 3 val
queries, canonical Swiss citation format normalization, end-to-end run under 11 h.

**Out of scope (later phases):** court corpus retrieval (Phase 2), entity-driven
direct lookup + citation graph (Phase 3), dense court reranking + jina cross-encoder
(Phase 4), Qwen LLM augmentation (Phase 5).

**Scope carve-out confirmed in discussion:** QUERY-01 (OpusMT EN→DE translation) is
officially mapped to Phase 2 in REQUIREMENTS.md, but is pulled forward into Phase 1
scope because the laws BM25 signal is unusable on English queries against a German
corpus. Plan should re-map QUERY-01 to Phase 1 when updating traceability.

</domain>

<decisions>
## Implementation Decisions

### Laws BM25 Query Language (the discussed area)

- **D-01:** **Pull OpusMT (`Helsinki-NLP/opus-mt-tc-big-en-de`) forward from Phase 2
  into Phase 1.** BM25 laws queries are the German translation of the English input,
  not the raw English text. Without translation, the sparse signal collapses to noise
  for any query that does not contain an explicit `Art. X` / law-code reference, which
  is most of val. The competitive cost of skipping translation here is larger than the
  scope cost of borrowing one requirement from Phase 2.

- **D-02:** **Translate once at startup, cache per query.** Load OpusMT, translate all
  val and test queries in a single batch, store `(query_id → german_translation)` in
  a Python dict, then **unload OpusMT and `cuda.empty_cache()` before BGE-M3 loads**
  (FOUND-03: one model on GPU at a time). Only ~50 queries total — the full translation
  step should take seconds.

- **D-03:** **Append regex-extracted legal codes to the translated query before BM25
  tokenization.** Keep the `extract_legal_codes()` helper from `solution.py` (Art./Abs.
  patterns, law abbreviations ZGB/OR/StGB/StPO/BGG/SchKG, SR numbers, BGE references).
  Extract from the **original English query** (codes are language-agnostic anyway), and
  concatenate them onto the German translation. Final BM25 input string =
  `{german_translation} {extracted_codes}`. This preserves the exact-match precision
  boost for queries that do have explicit references and layers it on top of real
  German lexical match for queries that don't.

- **D-04:** **Dense path is unchanged by this decision.** BGE-M3 dense retrieval for
  the laws corpus receives BOTH the raw English query and the German translation
  (per QUERY-03), embeds each, averages the two embeddings L2-normalized, and queries
  FAISS IndexFlatIP once. The translation is a free side product of the BM25 prep;
  reusing it on the dense side costs nothing and only adds signal.

- **D-05:** **Query canonicalization (QUERY-04) runs on both the English original and
  the German translation before downstream use.** Normalize whitespace around `Art.`
  and `Abs.`, normalize Roman / Arabic article numerals, normalize `ss` vs `ß`,
  normalize `ZGB`/`CC` aliasing. Applied before code extraction, before BM25
  tokenization, and before dense encoding. Same canonicalizer used for laws text
  preprocessing (LAWS-05) and for final submission citation formatting (CALIB-02) so
  the pipeline is self-consistent end to end.

### Model Lifecycle (enforced by FOUND-03)

- **D-06:** **Strict single-GPU-resident model invariant.** Order:
  `(1) load OpusMT → translate all queries → del + empty_cache`,
  `(2) load BGE-M3 fp16 → encode laws corpus → build FAISS → encode queries →
   kNN → del + empty_cache`,
  `(3) load cross-encoder → rerank → del + empty_cache`. Each stage writes to an
  intermediate dict/array so no neural model ever coexists with another on GPU.

- **D-07:** **CUDA assert is the very first executable cell** (FOUND-01/AH-1).
  `assert torch.cuda.is_available()`, log `torch.cuda.get_device_name(0)`, log VRAM
  total. Additionally, add a 60-second encoding checkpoint: if fewer than 1000 laws
  docs have been encoded after 60 s of wall-clock, raise and abort — catches the
  "CUDA present but kernel miscompiled" silent-CPU fallback that the team hit before.

### Claude's Discretion

The user delegated remaining choices ("select the best, I just want to win the
competition"). The planner / executor has flexibility on:

- **Notebook vs `solution.py` source of truth.** Default: refactor
  `notebook_kaggle.ipynb` in place as the canonical Kaggle-side artifact; keep
  `solution.py` as a local dev mirror only for fast local iteration. No requirement
  to keep the two byte-identical.
- **Cross-encoder choice for Phase 1.** Default: start with the existing
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (already in `models/` and in
  `solution.py`) to keep Phase 1 lightweight. Jina v2 is locked for Phase 4 by
  FUSE-03; swap there, not here.
- **Calibration approach.** Default: start with the existing `calibrate_top_k()`
  grid-search over K ∈ [1, 80] on val (solution.py lines 303-313). The 10-query val
  set is too small for aggressive per-query threshold tuning (CP-4 overfitting risk);
  LLM-based per-query K estimation is already scoped for Phase 5 (LLM-04).
- **FAISS install path.** Default: try `faiss-gpu-cu12` first; fall back to
  `faiss-cpu` on install failure. 175K docs in IndexFlatIP is fine on CPU if needed.
- **BGE-M3 laws K, BM25 laws K, RRF k constant, rerank pool size.** Start from
  `solution.py` defaults (`BM25_LAWS_K=100`, `DENSE_LAWS_K=100`, `RRF_K_CONST=60`,
  `RERANK_K=150`); tune on val only.
- **Legal code extraction regex coverage.** Extend the existing
  `extract_legal_codes()` set as needed; adding law codes is safe.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level specs
- `.planning/PROJECT.md` — overall goal, constraints, validated vs active requirements
- `.planning/REQUIREMENTS.md` — v1 requirement list; Phase 1 owns FOUND-01..06,
  LAWS-01..05, QUERY-03, QUERY-04, CALIB-01..05, INC-01..03 (and QUERY-01 pulled
  forward per D-01)
- `.planning/ROADMAP.md` §Phase 1 — phase goal, depends-on, success criteria

### Research artifacts
- `.planning/research/SUMMARY.md` — six-stage pipeline overview, stack rationale,
  expected F1 trajectory per phase
- `.planning/research/STACK.md` — BGE-M3 / bm25s / faiss-gpu-cu12 / OpusMT
  justifications and version pins
- `.planning/research/PITFALLS.md` — AH-1 (silent CPU fallback), AH-3 (fixed count),
  AH-4 (e5-small recall), CP-1 (German compound blindness), CP-3 (T4 bfloat16 /
  Flash Attention), CP-4 (macro-F1 overfitting on small val), CP-6 (Swiss citation
  format normalization)
- `.planning/research/FEATURES.md` — must-have table stakes vs competitive
  differentiators

### Codebase maps
- `.planning/codebase/ARCHITECTURE.md` — six-layer pattern and current component
  locations in `solution.py`
- `.planning/codebase/STRUCTURE.md` — directory layout, Kaggle path conventions,
  where to add new code
- `.planning/codebase/CONVENTIONS.md` — naming / logging / function-design patterns
  to keep the notebook consistent with existing code
- `.planning/codebase/STACK.md` — currently-installed library versions and their
  exact use sites in `solution.py`

### Existing code (must be read before refactoring)
- `solution.py` — lines 20-35 (Kaggle/local path config), 66-76 (`load_data`),
  82-89 (`tokenize_for_bm25` — preserve legal notation), 100-114 (BM25 build),
  120-166 (dense model + FAISS), 172-189 (OpusMT — reuse for D-01 translation),
  194-214 (reranker), 220-282 (`retrieve_candidates` + RRF), 288-313 (F1 +
  `calibrate_top_k`), 319-330 (`extract_legal_codes` — reuse for D-03)
- `notebook_kaggle.ipynb` — current Kaggle notebook (CPU, e5-small, laws-only);
  target artifact for Phase 1 refactor
- `download_models.py` — HF Hub download pattern; update to include BGE-M3 and
  OpusMT for Kaggle dataset packaging
- `requirements.txt` — update to add `bm25s`, `FlagEmbedding` (or transformers-only
  BGE-M3 load), `faiss-gpu-cu12` (with faiss-cpu fallback)

### Competition reference
- `Overview.md` — competition rules, submission format
- `AboutData.md` — data file schema (query_id, gold_citations, citation, text)
- `SETUP.md` — current local setup; update after Phase 1

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`tokenize_for_bm25(text)` (solution.py:82-89)** — regex-based tokenizer that
  preserves legal notation (`Art.`, `Abs.`, numbers). Reuse as-is for bm25s input;
  bm25s accepts pre-tokenized lists via `tokenize=lambda x: x.split()` or custom
  callables.
- **`extract_legal_codes(text)` (solution.py:319-330)** — pulls `(law_code,
  article_number)` from query text. Reuse directly for D-03 (append to translated
  German query before BM25 tokenization). Phase 3 will extend this same function
  for ENT-01..05.
- **`mean_pool(hidden, mask)` (solution.py:127-129)** — BGE-M3 is loaded with
  transformers and mean-pooled the same way; this helper transfers cleanly. BGE-M3
  uses `"query: "` prefix for queries and `"passage: "` for documents — distinct
  from e5, must update.
- **`rrf_fuse(ranked_lists, k=60)` (solution.py:220-226)** — already implements
  standard RRF fusion; works unchanged for the Phase 1 two-list case (laws-dense +
  laws-BM25) and for the later 3-5 list cases in Phases 2-4.
- **`calibrate_top_k(val_results)` (solution.py:303-313)** — grid search 1-80 on
  val macro-F1. Reuse for D-05 default calibration.
- **`macro_f1(gold_sets, pred_sets)` (solution.py:288-300)** — per-query F1
  averaging matching competition metric. Reuse for val logging (CALIB-04).
- **OpusMT load/translate/unload** (solution.py:172-189) — the exact pattern
  needed for D-01/D-02, just needs to be moved earlier in the pipeline and
  explicitly `del + empty_cache` after use.
- **Kaggle/local path detection** (solution.py:20-35) — `IS_KAGGLE` flag +
  conditional `DATA_DIR` / `E5_DIR` / `RERANK_DIR`. Extend with `BGE_M3_DIR` and
  `OPUS_MT_DIR` for Phase 1.

### Established Patterns
- **Monolithic script / single notebook, no package layering.** solution.py and
  `notebook_kaggle.ipynb` both keep helpers and main flow in one file. Phase 1
  should follow — don't introduce a `src/` package or import layering; the Kaggle
  notebook should be readable top-to-bottom as cells.
- **All-caps module-level constants** (`BM25_LAWS_K`, `RRF_K_CONST`,
  `RERANK_K`, `IS_KAGGLE`) at top of file. Phase 1 should add `BGE_M3_DIR`,
  `OPUS_MT_DIR`, `DENSE_LAWS_K`, `USE_RERANKER`, `USE_LLM_AUGMENTATION=False` in
  the same style — later phases will flip those flags without touching logic.
- **`print(..., flush=True)` logging with timing** (`time.time() - t0`) and
  section banners `print("\n=== Section Name ===")`. Phase 1 smoke test and
  pipeline logging should use this exact style so logs diff cleanly across phases.
- **Assertion-based validation, no try/except** (`assert DATA_DIR is not None`).
  Phase 1 should assert CUDA, assert embedding dim == 1024 after BGE-M3 load,
  assert submission.csv row count == test query count before writing.

### Integration Points
- **`solution.py::run()` / `notebook_kaggle.ipynb` last cell** — existing end-to-end
  entry point. Phase 1 replaces this with the BGE-M3 + bm25s + OpusMT + mmarco
  pipeline; the name and location stay the same so the watch scripts
  (`watch_and_submit.sh`, `watch_simple.sh`) still trigger correctly.
- **`Data/` (local) ↔ `/kaggle/input/llm-agentic-legal-information-retrieval/`
  (Kaggle)** — handled by IS_KAGGLE path switch; no new logic needed.
- **`models/` (local) ↔ `/kaggle/input/{dataset-slug}/` (Kaggle)** — Phase 1
  must ensure BGE-M3 and OpusMT are uploaded as Kaggle datasets before the
  notebook can be submitted. `download_models.py` is the local prep step.
- **`submission.csv`** — must land at `./submission.csv` (local) or
  `/kaggle/working/submission.csv` (Kaggle); CALIB-03 format validation runs
  immediately before writing.

</code_context>

<specifics>
## Specific Ideas

- **User's explicit priority signal:** "select the best, I just want to win the
  competition." Interpret as: optimize for Val Macro-F1 first, phase-scope purity
  second. Pulling QUERY-01 (translation) forward from Phase 2 is justified under
  this priority and was confirmed by the user.
- **Current baseline to beat:** Val Macro-F1 = 0.009 (effectively zero). Phase 1
  target is a meaningful laws-only F1 — research expects laws-only recall ceiling
  of ~59% (remaining 41% is court, Phase 2). Any F1 that represents a real cross-
  lingual retrieval rather than noise is a Phase 1 win.
- **Hard fail-loudly philosophy on Kaggle.** AH-1 (silent CPU fallback consumed
  1.7 h encoding time before the team noticed) drives FOUND-01 and the 60-second
  encoding checkpoint in D-07. Everything in Phase 1 that can assert, should assert.
- **No new architectural abstractions.** The existing six-layer structure in
  solution.py is the target shape; Phase 1 is a stack swap (e5-large → BGE-M3,
  rank-bm25 → bm25s, CPU → GPU) plus adding D-01's translation step, not a rewrite.

</specifics>

<deferred>
## Deferred Ideas

- **BGE-M3 built-in sparse head as primary lexical signal.** Rejected for Phase 1
  because it contradicts LAWS-03's explicit German decompounding requirement and
  has no published benchmark on Swiss legal corpora. Could be revisited in a v2
  milestone as an alternative to bm25s if BM25 laws recall underperforms.
- **Per-query cross-encoder score threshold for calibration.** Rejected for Phase 1
  because the 10-query val set is too small to fit per-query thresholds without
  overfitting (CP-4). LLM-based per-query K estimation is already scoped for
  Phase 5 (LLM-04), which is a more principled route.
- **Skipping BM25 laws entirely and going dense-only.** Rejected because it throws
  away guaranteed-precision exact-match hits on queries with explicit law codes.
- **Jumping directly to jina-reranker-v2 in Phase 1.** Deferred to Phase 4 per
  FUSE-03 to keep Phase 1's model count and upload surface minimal; the existing
  mmarco-mMiniLMv2 is sufficient for a laws-only baseline.

*No cross-referenced todos found.*

</deferred>

---

*Phase: 01-foundation-laws-pipeline*
*Context gathered: 2026-04-11*
