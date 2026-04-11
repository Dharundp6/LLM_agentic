---
phase: 01-foundation-laws-pipeline
plan: 04
subsystem: bm25s-laws-lexical
tags: [notebook, bm25s, german-tokenization, sparse-retrieval, laws-03, charsplit]
requires:
  - Plan 01-01 Cells 0-3 (tokenize_for_bm25_de, canonicalize, laws DataFrame, BM25_LAWS_K, SMOKE_LAWS_N, time import)
  - Plan 01-02 Cell 4 (bm25_query_texts dict — '{de_canon} {codes}' per D-03)
  - Plan 01-03 Cell 7 (_target_ids list — val + test query IDs already defined)
  - bm25s>=0.2.0 on PYTHONPATH (Kaggle runtime install from Cell 0)
provides:
  - notebook_kaggle.ipynb Cell 8 — bm25s laws index build + per-query BM25 retrieval
  - bm25_laws — bm25s.BM25() index over tokenize_for_bm25_de(laws citation + text) tokens
  - bm25_laws_ids[qid] -> list[int] of top BM25_LAWS_K laws row indices for every val+test query
  - Inline 'Art.' survival assertion guarding LAWS-03 against tokenizer regression
affects:
  - Plan 01-05 (RRF fusion): imports bm25_laws_ids as the BM25 recall signal alongside dense_laws_ids
  - Phase 2 court BM25: Cell 8 establishes the bm25s pre-tokenized list-of-list pattern reused at 2.47M scale
tech-stack:
  added:
    - bm25s.BM25 (bm25s already pinned in requirements.txt / installed in Cell 0 from Plan 01-01)
  patterns:
    - Research Pattern 5 (bm25s with pre-tokenized list-of-list input, no built-in bm25s.tokenize())
    - Research Common Pitfalls CP-1 (NO Snowball stemming wired into bm25s)
    - Research Common Pitfalls Pitfall 5 ('Art.' survival assertion on a representative query)
    - Idempotent nbformat builder script convention (matches Plans 01-01 / 01-02 / 01-03)
key-files:
  created:
    - path: scripts/build_notebook_cell_8.py
      role: Idempotent nbformat editor that installs Cell 8 in place; dependency-checks Cells 3/4/7
    - path: .planning/phases/01-foundation-laws-pipeline/01-04-SUMMARY.md
      role: This summary
  modified:
    - path: notebook_kaggle.ipynb
      role: Cell 8 appended — bm25s laws index build + per-query BM25 retrieval
decisions:
  - D-03 enacted in Cell 8: per-query BM25 retrieval uses bm25_query_texts[qid] (German canon + extracted English legal codes) as the input text, tokenized through the Cell 3 tokenize_for_bm25_de CharSplit-aware pipeline
  - LAWS-03 enforced: Cell 8 source contains no 'stemmer' token anywhere (comment phrased as "NO Snowball stemming" after acceptance-check rejection of literal 'stemmer')
  - Pre-tokenized list-of-list is the bm25s input — bm25s.tokenize() is explicitly avoided so CharSplit decompounding + NLTK German stopword filtering both apply
  - Pitfall 5 mitigated inline: tokenize_for_bm25_de("Die Pflicht aus Art. 41 OR") is asserted to contain 'art.' before indexing, so a regex/stopword regression fails the cell before the 175K-doc index is built
  - Idempotent builder script matches Plan 01-01 / 01-02 / 01-03 convention (checked-in, dependency-guarded, re-runnable)
  - SMOKE mode path is inherited from Cell 6 (laws is already truncated to SMOKE_LAWS_N rows by Plan 01-03 Cell 6 when SMOKE is True); Cell 8 does not re-apply SMOKE truncation because it would be redundant and could desync laws row indices from the FAISS index
metrics:
  duration_minutes: 4
  completed_date: "2026-04-11"
  tasks_completed: 1
  files_created: 2
  files_modified: 1
requirements_satisfied:
  - LAWS-03 (German decompounding via CharSplit; NO Snowball stemming; NLTK German stopwords)
  - LAWS-04 (per-query BM25 retrieval populates bm25_laws_ids for every val + test query)
  - FOUND-06 (bm25s retrieval stage logged with elapsed timer; bounded work per query)
---

# Phase 1 Plan 04: bm25s Laws Lexical Retrieval Summary

Wire bm25s as the laws corpus lexical retriever: tokenize the (already-canonicalized) laws corpus through `tokenize_for_bm25_de` (CharSplit + NLTK German stopwords, no stemming), build a `bm25s.BM25()` index from the pre-tokenized list-of-list input, then run per-query top-`BM25_LAWS_K` retrieval using `bm25_query_texts[qid]` (D-03: German translation + English-extracted legal codes). Populate `bm25_laws_ids` for every val + test query. This closes the BM25 half of the laws recall plane and hands the lexical signal to the RRF fusion plan downstream.

## Context

Plan 01-03 left the notebook with Cells 0-7: CUDA gate, path constants, canonicalization helpers, OpusMT translation, BGE-M3 laws dense retrieval, and BGE-M3 unload with VRAM guard. The dense signal is already in `dense_laws_ids[qid]`. Plan 01-04 adds the lexical signal in a single Cell 8.

Getting BM25 on German laws right is load-bearing for queries that carry explicit legal code references like "Art. 41 OR" — dense retrieval matches the semantic content, but exact code-number patterns are a BM25 strength and a core reason the project uses both. Two things sink BM25 on this corpus and both are called out in the research `01-RESEARCH.md`:

1. **CP-1**: wiring Snowball into bm25s collapses `art.` into `art`, `arti`, etc., destroying the legal-notation tokens the laws corpus relies on. Cell 8's source is scrubbed of the literal `stemmer` token to fail the plan acceptance check loudly if anyone later reintroduces it.
2. **Pitfall 5**: a tokenizer regression that drops `Art.` silently slashes BM25 recall without any runtime error. Cell 8 asserts `"art." in tokenize_for_bm25_de("Die Pflicht aus Art. 41 OR")` inline, before the 175K-doc index build, so a broken tokenizer fails the cell in under a second instead of poisoning downstream plans.

One task in one commit.

## Key Changes

### notebook_kaggle.ipynb — Cell 8 (appended)

Appended as Cell 8 (notebook now has 9 cells total). The source does the following:

- **Import**: `import bm25s` scoped to this cell — not top-of-notebook. Plan 01-01 already installed `bm25s>=0.2.0` in Cell 0's Kaggle pip block.
- **Tokenize laws corpus**: builds `laws_bm25_texts = [f"{row['citation']} {row['text']}" for _, row in laws.iterrows()]` then runs each text through `tokenize_for_bm25_de` (from Cell 3) to produce `laws_bm25_tokens: list[list[str]]`. `laws` was already canonicalized in place by Plan 01-01 Cell 3 (LAWS-05), so no re-canonicalization is needed here.
- **Pitfall 5 guardrail**: `_smoke_sample = tokenize_for_bm25_de("Die Pflicht aus Art. 41 OR")` then `assert "art." in _smoke_sample`. This is a one-line canary that fires before the index build. Cell 3 already has a similar assertion on the tokenizer definition; Cell 8 repeats it so a later edit to Cell 3 that breaks the tokenizer is caught at the point of use as well as the point of definition.
- **Build index**: `bm25_laws = bm25s.BM25()` then `bm25_laws.index(laws_bm25_tokens)`. This is Research Pattern 5 verbatim — the list-of-list pre-tokenized input path. No `stemmer=` argument is passed (CP-1). No `bm25s.tokenize()` call is used (that built-in bypasses the CharSplit decompounding wired into `tokenize_for_bm25_de`).
- **Per-query retrieval**: loops over `_target_ids` (defined in Cell 7, honors SMOKE_VAL_N for val and always uses full test). For each qid, the input text is `bm25_query_texts[qid]` — the D-03 format `"{de_canon} {codes}"` built in Cell 4. Tokens come from the same `tokenize_for_bm25_de` to ensure query/corpus tokenization symmetry (which is the most common accidental cause of BM25 recall collapse). Empty-token defense: `if not q_tokens: bm25_laws_ids[qid] = []; continue`.
- **Result shape normalization**: `results, scores = bm25_laws.retrieve([q_tokens], k=BM25_LAWS_K)` then `bm25_laws_ids[qid] = [int(x) for x in results[0]]`. bm25s returns numpy int arrays from `retrieve()`; the explicit `int(x)` cast normalizes across bm25s 0.2.x / 0.2.5+ minor versions where the array dtype has varied.
- **Sanity log** on the first query: `_target_ids[0]` and `bm25_laws_ids[_target_ids[0]][:5]` — the top-5 BM25 laws row indices for the first val query. Plan 01-05's RRF fusion will consume these identifiers directly.

### scripts/build_notebook_cell_8.py — new

Idempotent nbformat editor matching the Plan 01-01 / 01-02 / 01-03 convention. Hard-fails with distinct exit codes if any of the dependency cells is missing its contract:

- Exit 2: notebook has fewer than 8 cells (Plans 01-01..01-03 not applied).
- Exit 3: Cell 3 missing `def tokenize_for_bm25_de` (Plan 01-01 not applied).
- Exit 4: Cell 4 missing `bm25_query_texts` (Plan 01-02 not applied).
- Exit 5: Cell 7 missing `_target_ids` (Plan 01-03 not applied).

On the success path, the script replaces Cell 8 in place if one already exists, otherwise appends. `id` is popped from the freshly built cell so `nbformat_minor=4` reads cleanly (matches Plans 01-02 / 01-03 — the pre-existing minor-version mismatch from Plan 01-01 is still deferred, see Deferred Issues below).

## Commits

| Task | Hash | Message |
|------|------|---------|
| Task 1 | `fc89b6d` | feat(01-04): add notebook Cell 8 with bm25s laws index + per-query BM25 retrieval |

## Verification Results

All plan acceptance criteria pass on the final notebook:

```
OK - all plan acceptance criteria pass
Total cells: 9
Cell 8 len: 1958 chars
```

Task 1 acceptance criteria coverage (nine explicit checks from the plan):

1. `len(nb.cells) >= 9` — notebook has 9 cells.
2. `'import bm25s' in s` — import present.
3. `'bm25s.BM25()' in s` — index constructor present.
4. `'bm25_laws.index(laws_bm25_tokens)' in s` — index build call present.
5. `'tokenize_for_bm25_de' in s` — Cell 3 tokenizer reused.
6. `'bm25_query_texts[qid]' in s` — D-03 per-query source present.
7. `'bm25_laws_ids' in s and 'bm25_laws.retrieve([q_tokens], k=BM25_LAWS_K)' in s` — retrieval loop present.
8. `'stemmer' not in s` — CP-1 forbidden token absent (see Deviations).
9. `'assert "art." in _smoke_sample' in s` — Pitfall 5 survival assertion present.

Additional self-checks beyond the plan's acceptance list:

- Cell 8 source parses as valid Python via `ast.parse()`.
- `scripts/build_notebook_cell_8.py` parses as valid Python via `ast.parse()`.
- Cells 0-7 are byte-intact (spot-checked `IS_KAGGLE` in Cell 0, `def unload` in Cell 1, `laws_de.csv` in Cell 2, `def tokenize_for_bm25_de` in Cell 3, `MarianMTModel` in Cell 4, `def bge_encode` in Cell 5, `faiss.IndexFlatIP(1024)` in Cell 6, `_target_ids` in Cell 7).
- `nbformat.read('notebook_kaggle.ipynb', 4)` succeeds on the final notebook.
- Commit `fc89b6d` exists on branch `worktree-agent-a00b9038`, staged exactly `notebook_kaggle.ipynb` and `scripts/build_notebook_cell_8.py`.
- Commit message contains no AI attribution per CLAUDE.md rule.

Runtime execution of Cell 8 (actual bm25s index build over 175K laws docs + per-query retrieval) is deferred to Kaggle. No local copy of `laws_de.csv` is present in the worktree, and bm25s is not installed in the local Python environment. This matches the plan's `<verification>` note: "nbformat check in Task 1 verifies cell presence." Plan 01-05's RRF fusion is the first cell that exercises `bm25_laws_ids` end-to-end; runtime integration verification will happen there.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stripped the literal `stemmer` token from Cell 8 source**

- **Found during:** Task 1 acceptance verification.
- **Issue:** The plan's `<action>` text specified a comment reading `# NO Snowball stemmer. Pre-tokenized list-of-list input.` but the plan's `<acceptance_criteria>` simultaneously asserted `assert 'stemmer' not in s`. These two clauses contradict: the comment itself contains the token the assertion forbids. Writing the plan verbatim would cause the acceptance check to fail.
- **Fix:** Rephrased the comment to `# NO Snowball stemming (PITFALLS CP-1). Pre-tokenized list-of-list input.` — same semantic content (no Snowball stemming, CP-1 compliance), but the token `stemmer` no longer appears anywhere in the cell source. The CP-1 reference is also tightened (it now explicitly points to the pitfalls identifier, which is arguably clearer than the original plan text).
- **Files modified:** `scripts/build_notebook_cell_8.py` (the builder script's embedded CELL_SOURCE triple-string), which then rewrote `notebook_kaggle.ipynb` Cell 8 on re-run.
- **Commit:** `fc89b6d` (the fix was caught before the first commit, so the committed source already has the stripped token).
- **Why this is Rule 1 not Rule 4:** The acceptance criterion is the plan's own success gate. If the gate says "no stemmer token" and the body says "include a comment with the stemmer token", the gate takes precedence — it is the only programmatic check the plan defines. No architectural change, no scope expansion, just a one-word comment rephrase to keep the forbidden token out of the source while preserving the intended message. Tracked here so the verifier sees the intentional divergence from the plan's verbatim action text.

### Scope clarifications (not Rule 1/2/3 auto-fixes)

- **Builder script instead of inline `python -c`.** The plan's action text could in principle have been an inline nbformat edit. The checked-in `scripts/build_notebook_cell_8.py` matches the convention from Plans 01-01, 01-02, and 01-03 (each of those plans' summaries explicitly calls this out as a scope clarification). Committed alongside the notebook change.
- **Cell 8 appended, not replaced.** The notebook started at exactly 8 cells after Plan 01-03, so a literal "append" at index 8 is correct and matches the plan's `<acceptance_criteria>` which check `nb.cells[8].source`. The builder script is self-healing against either starting state (append if `len(nb.cells) < 9`, replace in place otherwise).
- **Cell 8 does not re-apply the SMOKE truncation to `laws`.** Plan 01-03 Cell 6 already truncates `laws` to `SMOKE_LAWS_N` rows when `SMOKE=True` (as part of the dense encoding stage), and that truncation persists into the Cell 8 world — `laws` is the same DataFrame object. Re-applying `laws.iloc[:SMOKE_LAWS_N]` in Cell 8 would be a no-op in the smoke path but would risk desyncing `laws` row indices between the FAISS index (built in Cell 6) and the BM25 index (built here in Cell 8), which would silently break RRF fusion in Plan 01-05. Leaving the truncation single-sourced in Cell 6 is the safer invariant.

## Known Stubs

None introduced by this plan. Cell 8 is fully implemented end-to-end: tokenize → index → per-query retrieve → log.

## Threat Flags

None. All three STRIDE entries in the plan's `<threat_model>` are accounted for:

- **T-01-04-01 (Tampering on tokenize_for_bm25_de correctness)** — mitigate. Inline `'art.' in _smoke_sample` assertion fires before the 175K-doc index build.
- **T-01-04-02 (DoS via index RAM)** — accept. 175K laws docs is well within the 30GB Kaggle budget; Phase 2 court BM25 (2.47M docs) is the real RAM risk, not this plan.
- **T-01-04-03 (Information Disclosure)** — accept. Index is ephemeral in-process; no writes to disk.

No new network endpoints, auth paths, file-access patterns, or schema changes introduced at trust boundaries beyond what the threat register already documented.

## Deferred Issues

- **Pre-existing `nbformat_minor` mismatch** carried from Plans 01-02 and 01-03. The notebook declares `nbformat_minor=4` but Cells 0-3 (written by Plan 01-01) contain `id` fields that only exist in minor=5+. `nbformat.validate()` raises, but `nbformat.read(..., 4)` — which every plan acceptance check uses — works. Plan 01-04's new builder script pops `id` from the freshly built Cell 8, matching Plans 01-02 / 01-03. Out of scope per Rule 3 scope-boundary; still filed for a later cleanup pass.

## Files Touched

| File | Status | Purpose |
|------|--------|---------|
| `notebook_kaggle.ipynb` | modified | Cell 8 appended — bm25s laws index + per-query BM25 retrieval |
| `scripts/build_notebook_cell_8.py` | created | Idempotent nbformat editor for Cell 8 (matches 01-01..01-03 convention) |
| `.planning/phases/01-foundation-laws-pipeline/01-04-SUMMARY.md` | created | This summary |

## Next Steps (Plan 01-05)

Plan 01-05 will RRF-fuse the BM25 and dense signals and run the mmarco cross-encoder over the fused pool:

- `bm25_laws_ids[qid]` (from this plan) and `dense_laws_ids[qid]` (from Plan 01-03) will be the two input rank lists to `rrf_fuse(k=RRF_K_CONST)`.
- `q_en_canon[qid]` (from Plan 01-02 Cell 4) will be the query side of the cross-encoder pair.
- The VRAM-free budget after Plan 01-03's Cell 7 unload is at least 10 GB, which is the gate that lets `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` load cleanly.

## Self-Check: PASSED

Verified all created files exist:

- `notebook_kaggle.ipynb` — FOUND (9 cells; Cell 8 contains `import bm25s`, `bm25s.BM25()`, `bm25_laws.index(laws_bm25_tokens)`, `tokenize_for_bm25_de`, `bm25_query_texts[qid]`, `bm25_laws_ids`, `bm25_laws.retrieve([q_tokens], k=BM25_LAWS_K)`, `assert "art." in _smoke_sample`; does NOT contain the literal `stemmer` token)
- `scripts/build_notebook_cell_8.py` — FOUND (Python AST valid; idempotent; dependency-checks Cells 3/4/7)
- `.planning/phases/01-foundation-laws-pipeline/01-04-SUMMARY.md` — FOUND (this file)

Verified the task commit exists on the worktree branch:

- `fc89b6d` — FOUND (feat(01-04): add notebook Cell 8 with bm25s laws index + per-query BM25 retrieval)

Plan acceptance criteria — all nine pass (see Verification Results section above).
