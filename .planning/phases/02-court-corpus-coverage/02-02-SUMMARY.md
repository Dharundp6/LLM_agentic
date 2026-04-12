---
phase: 02-court-corpus-coverage
plan: 02
subsystem: rrf-fusion-dual-corpus
tags: [rrf-fusion, cross-encoder, calibration, submission, court-integration, court-id-offset]
dependency_graph:
  requires: [02-01]
  provides: [fused_ids, reranked_ids, resolve_citation, resolve_text, COURT_ID_OFFSET]
  affects: [submission.csv, val-f1-calibration, pipeline-summary]
tech_stack:
  added: []
  patterns: [court-id-offset-namespace, dual-corpus-resolve-helpers, 3-signal-rrf]
key_files:
  created: []
  modified: [notebook_kaggle.ipynb]
decisions:
  - "COURT_ID_OFFSET = len(laws) namespaces court IDs to prevent collision with laws row indices"
  - "court_de_texts extracted in Cell 9 alongside court_de_citations for cross-encoder text access"
  - "resolve_citation() and resolve_text() centralize dual-corpus ID resolution"
  - "Variable rename from fused_laws_ids/reranked_laws_ids to fused_ids/reranked_ids for consistency"
metrics:
  duration: "3m 56s"
  completed: "2026-04-12T16:55:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 1
---

# Phase 2 Plan 2: 3-Signal RRF Fusion + Dual-Corpus Pipeline Summary

End-to-end court integration via COURT_ID_OFFSET namespace in RRF fusion, cross-encoder rerank on fused pool, and dual-corpus citation resolution for calibration and submission output.

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Cell 11 RRF 3-signal fusion + Cell 9 court_de_texts | `6ea0ab1` | notebook_kaggle.ipynb (Cell 9 + Cell 11) |
| 2 | Cells 12-15 dual-corpus cross-encoder, calibration, submission | `db3f03c` | notebook_kaggle.ipynb (Cells 12-15) |

## Changes Made

### Cell 9 Update (from Plan 01)
- Added `court_de_texts = court_de["text"].tolist()` after `court_de_citations` extraction
- Preserves court document texts in memory (~1.5 GB) for cross-encoder reranking
- Added `court_de_texts = []` in the `USE_COURT_CORPUS=False` else branch

### Cell 11: 3-Signal RRF Fusion (REPLACED)
- Defined `COURT_ID_OFFSET = len(laws)` to namespace court row indices
- `rrf_fuse()` function unchanged (already handles arbitrary signal count)
- For each query, builds 3 signal lists: dense_laws_ids, bm25_laws_ids, court-BM25 (offset)
- Court IDs offset by `+ COURT_ID_OFFSET` before fusion to prevent collision
- `resolve_citation(fused_id)`: maps namespaced ID to canonical citation string
- `resolve_text(fused_id, max_len=1500)`: maps namespaced ID to document text for cross-encoder
- Logs fusion pool composition (laws vs court IDs)
- Output: `fused_ids` dict (replaces `fused_laws_ids`)

### Cell 12: Cross-Encoder Rerank (MODIFIED)
- Uses `fused_ids` instead of `fused_laws_ids`
- Uses `resolve_text(idx)` for candidate text construction (both laws and court)
- Output: `reranked_ids` dict (replaces `reranked_laws_ids`)
- Logs reranked pool composition (laws vs court candidates)
- D-14: cross-encoder now reranks the FUSED pool containing both corpora

### Cell 13: Val F1 + Calibration (MODIFIED)
- Uses `reranked_ids` instead of `reranked_laws_ids`
- Uses `resolve_citation(idx)` for citation string resolution (dual-corpus)
- Prints D-12 completion line: `fused-3sig: F1=X.XXXX @ top-K`
- Adds Phase 2 success check message

### Cell 14: Submission Write (MODIFIED)
- Uses `reranked_ids` instead of `reranked_laws_ids`
- Uses `resolve_citation(idx)` for test submission citations
- Court citations (BGE/BGer docket format) now appear in submission.csv

### Cell 15: Pipeline Summary (MODIFIED)
- Logs court corpus size and signal count (3 or 2)
- Updated exit banner for Phase 2

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Functionality] court_de_texts extraction in Cell 9**
- **Found during:** Task 1
- **Issue:** Plan correctly identified that Cell 9 deletes court_de DataFrame after extracting citations, but cross-encoder needs document texts
- **Fix:** Added `court_de_texts = court_de["text"].tolist()` in Cell 9 before DataFrame deletion
- **Files modified:** notebook_kaggle.ipynb (Cell 9)
- **Commit:** 6ea0ab1

## Known Stubs

None. All code paths are fully wired with data sources. resolve_citation() and resolve_text() handle both laws and court IDs through the COURT_ID_OFFSET namespace.

## Verification Results

1. Notebook has 16 cells (unchanged count)
2. No cell 11+ contains `fused_laws_ids` or `reranked_laws_ids` (old variable names)
3. Cell 11 defines COURT_ID_OFFSET, resolve_citation(), resolve_text()
4. Cell 12 uses resolve_text() for cross-encoder candidate text
5. Cell 13 uses resolve_citation() for val calibration, prints fused-Nsig F1
6. Cell 14 uses resolve_citation() for submission writing
7. Cell 15 logs court corpus stats
8. Notebook JSON is valid (nbformat.read succeeds)

## Self-Check: PASSED

- notebook_kaggle.ipynb: FOUND
- 02-02-SUMMARY.md: FOUND
- Commit 6ea0ab1: FOUND
- Commit db3f03c: FOUND
