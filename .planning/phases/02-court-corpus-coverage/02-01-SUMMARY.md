---
phase: 02-court-corpus-coverage
plan: 01
subsystem: court-bm25-retrieval
tags: [court-corpus, bm25s, german-filter, ram-profiling, retrieval]
dependency_graph:
  requires: [01-foundation-laws-pipeline]
  provides: [bm25_court, court_de_citations, bm25_court_ids, per-signal-f1]
  affects: [RRF-fusion-cell-11, reranker-cell-12, calibration-cell-13]
tech_stack:
  added: []
  patterns: [chunked-tokenization-with-rss, german-stopword-heuristic, local-macro-f1-copy]
key_files:
  created: []
  modified: [notebook_kaggle.ipynb]
decisions:
  - "Local macro_f1 + _quick_f1 defined in Cell 10 to avoid forward reference to Cell 13"
  - "is_german() uses 50-word frozen set with 2-of-first-20 threshold; rejects <30 char stubs"
  - "resource module imported unconditionally; Windows fallback uses psutil with try/except"
metrics:
  duration: "3m 23s"
  completed: "2026-04-12T16:10:23Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 1
---

# Phase 2 Plan 1: Court BM25 Index + Retrieval Summary

Court corpus BM25 pipeline added as Cells 9-10 with German-only filtering, chunked tokenization with RSS profiling, and per-signal val F1 diagnostic logging.

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Cell 9: Court load + German filter + bm25s index | `769d6ac` | notebook_kaggle.ipynb (Cell 0 + Cell 9) |
| 2 | Cell 10: BM25 court retrieval + per-signal val F1 | `0468ceb` | notebook_kaggle.ipynb (Cell 10) |

## Changes Made

### Cell 0 Updates
- `USE_COURT_CORPUS` flipped from `False` to `True`
- `BM25_COURT_K = 200` added to hyperparameters section

### Cell 9: Court Corpus Load + German Filter + bm25s Index (NEW)
- `is_german()` heuristic: 50-word German stopword frozen set, checks first 20 words for >= 2 matches, rejects texts < 30 chars
- `log_rss()` helper: Linux uses `resource.getrusage()`, Windows uses `psutil`, warns at > 25 GB
- Loads `court_considerations.csv` with `fillna("")` on both columns
- German filter via `court["text"].apply(is_german)` with count/percentage logging (D-08)
- Tokenization in 500K-row chunks using array slicing (not iterrows) with RSS logged after each chunk (D-09)
- bm25s index build with immediate `del court_bm25_tokens; gc.collect()` to free ~11 GB (D-09)
- `court_de_citations` list extracted before `del court_de; gc.collect()`
- All code gated by `USE_COURT_CORPUS` feature flag

### Cell 10: BM25 Court Retrieval + Per-Signal Val F1 (NEW)
- Per-query BM25 court retrieval using `bm25_court.retrieve([q_tokens], k=BM25_COURT_K)` (D-10)
- Per-query timing with 2s warning threshold logged
- Reuses `bm25_query_texts[qid]` from Cell 4 (D-13)
- Local `_macro_f1` and `_quick_f1` defined to avoid forward reference to Cell 13
- Per-signal val F1 comparison table for laws-dense, laws-BM25, court-BM25 (D-12/FUSE-05)
- `bm25_court_ids` dict available for downstream RRF fusion
- Feature flag gated on `USE_COURT_CORPUS and bm25_court is not None`

### Downstream Cell Renumbering
- Former Cells 9-13 are now Cells 11-15 (shifted +2 by insertions)
- No content changes to shifted cells; header comments still reference old numbers (cosmetic only)

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None. All code paths are fully wired with data sources.

## Verification Results

1. Notebook has 16 cells (was 14)
2. Cell 0 contains `USE_COURT_CORPUS = True` and `BM25_COURT_K = 200`
3. Cell 9 contains `is_german`, `log_rss`, court CSV load, German filter, chunked tokenization, bm25s index build, token cleanup
4. Cell 10 contains `bm25_court_ids`, `bm25_court.retrieve`, per-signal val F1 table
5. Both cells gated by `USE_COURT_CORPUS` feature flag
6. Notebook JSON valid (nbformat.read succeeds)

## Self-Check: PASSED

- notebook_kaggle.ipynb: FOUND
- 02-01-SUMMARY.md: FOUND
- Commit 769d6ac: FOUND
- Commit 0468ceb: FOUND
