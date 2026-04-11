---
phase: 01-foundation-laws-pipeline
plan: 05
subsystem: pipeline-closure
status: partial / checkpoint-pending
tags: [notebook, rrf, cross-encoder, calibration, submission, smoke-gate, human-verify, phase-1-tag]
requires:
  - Plan 01-01 Cells 0-3 (paths, DEVICE, canonicalize, USE_RERANKER, SMOKE, RERANK_K, RRF_K_CONST, OUT_PATH, unload)
  - Plan 01-02 Cell 4 (q_en_canon, translations, bm25_query_texts, all_query_ids)
  - Plan 01-03 Cell 7 (dense_laws_ids, _target_ids, BGE-M3 unload — D-06 precondition for reranker load)
  - Plan 01-04 Cell 8 (bm25_laws_ids)
  - solution.py:194-226 (rrf_fuse + cross-encoder reference port)
  - solution.py:288-313 (macro_f1 + calibrate_top_k reference port)
provides:
  - notebook_kaggle.ipynb Cell 9  — rrf_fuse helper + fused_laws_ids per query (RRF_K_CONST=60)
  - notebook_kaggle.ipynb Cell 10 — mmarco cross-encoder rerank + reranked_laws_ids (USE_RERANKER-gated, unloads after)
  - notebook_kaggle.ipynb Cell 11 — macro_f1, calibrate_top_k (k_max=60, val-only, CP-4 boundary warn), *** Val macro-F1 *** banner
  - notebook_kaggle.ipynb Cell 12 — validate_submission + canonicalized test predictions + submission.csv write
  - notebook_kaggle.ipynb Cell 13 — pipeline completion summary + SMOKE/FULL exit banner
affects:
  - Phase 1 submission: writes submission.csv to OUT_PATH (Kaggle: /kaggle/working/submission.csv; local: repo root)
  - Phase 2+ court pipeline: the RRF/rerank/calibrate/validate/write skeleton in Cells 9-13 is the pattern Phase 2 will extend to add the court recall plane
  - Phase 1 `phase-1-submission` git tag: created manually by user at the human-verify checkpoint (Task 4)
tech-stack:
  added:
    - transformers.AutoModelForSequenceClassification (mmarco-mMiniLMv2-L12-H384-v1 cross-encoder; already pinned in requirements.txt via Plan 01-01)
  patterns:
    - Research Pattern 6 (cross-encoder load AFTER BGE-M3 unload — D-06)
    - solution.py RRF fuse verbatim (lines 220-226)
    - solution.py macro_f1 + calibrate_top_k verbatim (lines 288-313) with k_max=60 (plan-level hyperparameter)
    - Research §Submission Validator (validate_submission asserts BEFORE to_csv — CALIB-03)
    - Idempotent nbformat builder script convention (matches Plans 01-01..01-04)
key-files:
  created:
    - path: scripts/build_notebook_cells_9_10.py
      role: Idempotent nbformat editor for Cells 9 (RRF fuse) and 10 (cross-encoder rerank + unload)
    - path: scripts/build_notebook_cells_11_12.py
      role: Idempotent nbformat editor for Cells 11 (calibrate) and 12 (validate + write submission)
    - path: scripts/build_notebook_cell_13.py
      role: Idempotent nbformat editor for Cell 13 (pipeline completion summary)
    - path: .planning/phases/01-foundation-laws-pipeline/01-05-SUMMARY.md
      role: This summary (partial / checkpoint-pending)
  modified:
    - path: notebook_kaggle.ipynb
      role: Cells 9-13 appended — closes the Phase 1 foundation + laws pipeline end-to-end
decisions:
  - D-06 re-verified at load site: Cell 10 loads mmarco cross-encoder ONLY after the Cell 7 unload(bge_tok, bge_mdl) has run. No reordering in this plan.
  - k_max=60 for calibrate_top_k (not solution.py's 80) — the plan-level hyperparameter fixed in Plan 01-01 Cell 0. Boundary WARN (CP-4) fires if best_k lands at k_min=1 or k_max=60.
  - CALIB-05 enforced: `calibrate_top_k(val_results, k_min=1, k_max=60)` receives ONLY val_results. Test predictions are generated in a separate cell (Cell 12) that applies best_k read-only. The val/test split is never crossed.
  - CALIB-02 (canonicalize) applied on the WRITE side in Cell 12: every laws.iloc[idx]['citation'] passed to submission_df.predicted_citations is routed through canonicalize() from Cell 3. Single source of truth for the canonical form.
  - CALIB-03 (validate_submission BEFORE to_csv) enforced: Cell 12 calls validate_submission(submission_df, test) before submission_df.to_csv(OUT_PATH). A bad submission fails loudly instead of silently clobbering the output file.
  - USE_RERANKER gate (INC-02): Cell 10 wraps the entire cross-encoder load/rerank/unload sequence in `if USE_RERANKER:` and falls back to `reranked_laws_ids[qid] = fused_laws_ids[qid][:RERANK_K]` in the disabled branch. Phase 1 ships with USE_RERANKER=True per Cell 0.
  - SMOKE gating is inline in Cells 6/7 (not Cell 13). Cell 13 is a terminal summary, not a smoke runner — by the time control reaches it, the end-to-end pipeline has already completed under whatever SMOKE flag is set in Cell 0.
  - Submission dedupe preserves order: Cell 12 uses a `seen` set to drop duplicate canonical citations from the top-k list before ';'.join(), which is safe because canonicalize() is the single source of truth for the match key.
  - Builder scripts split across three files (cells_9_10, cells_11_12, cell_13) — one per task — matching the one-task-one-commit Phase 1 convention (Plans 01-01..01-04 each have their own builder script).
metrics:
  duration_minutes: 8
  completed_date: "2026-04-11"
  tasks_completed: 3
  tasks_total: 4
  tasks_pending:
    - "Task 4: human-verify + `git tag phase-1-submission` (BLOCKED on user Kaggle run)"
  files_created: 4
  files_modified: 1
requirements_satisfied_structurally:
  - FOUND-03 (BGE-M3 unload precondition honored: Cell 10 cross-encoder load is sequenced AFTER Cell 7 unload)
  - FOUND-05 (SMOKE path end-to-end — structural only; runtime verification deferred to Kaggle in Task 4)
  - FOUND-06 (RRF + rerank stages logged with elapsed timers and bounded per-query work)
  - CALIB-01 (macro_f1 computed on val per-query, averaged)
  - CALIB-02 (canonicalize applied on write in Cell 12 before ';'.join)
  - CALIB-03 (validate_submission runs BEFORE to_csv in Cell 12)
  - CALIB-04 (calibrate_top_k grid search + boundary WARN present; best_k logged with banner)
  - CALIB-05 (calibrate_top_k receives val_results only; test is applied in a separate cell)
  - INC-01 (pipeline is a single notebook run Cells 0-13)
requirements_pending_human_verify:
  - INC-03 (`git tag phase-1-submission` is user-gated at the Task 4 checkpoint; agent will NOT run git tag autonomously per CLAUDE.md)
---

# Phase 1 Plan 05: Pipeline Closure Summary (partial / checkpoint-pending)

**Status: PARTIAL.** Tasks 1-3 auto-executed on the worktree branch; Task 4 is the blocking `checkpoint:human-verify` gate waiting on a Kaggle run + human inspection + manual `git tag phase-1-submission`. This summary is written now (on completion of the autonomous portion) and committed on the worktree branch so the orchestrator can reach it, but the final Task 4 confirmation is still outstanding.

## One-Liner

Cells 9-13 of `notebook_kaggle.ipynb` now RRF-fuse the laws dense + bm25 rank lists, rerank with the mmarco cross-encoder (AFTER BGE-M3 unload per D-06), calibrate best_k on val only (grid [1,60], CP-4 boundary warn), and validate-then-write `submission.csv` — closing the Phase 1 foundation + laws pipeline end-to-end. Human-verify on Kaggle + `git tag phase-1-submission` still pending.

## Context

Plan 01-04 left `notebook_kaggle.ipynb` at 9 cells (0-8), with both laws recall planes populated: `dense_laws_ids` from Cell 7 (BGE-M3 via FAISS) and `bm25_laws_ids` from Cell 8 (bm25s with CharSplit-aware German tokenization). What was missing was everything that turns two rank-index lists into a valid submission: fusion, reranking, calibration, validation, and the CSV write. Plan 01-05 is the Phase 1 payload — not because the individual pieces are novel, but because `calibrate_top_k(val_results)` touching test or `to_csv()` running before `validate_submission()` silently invalidates the submission. The plan is "compose existing helpers correctly, in the right order, with the right gates."

One task per cell pair, one commit per task, one builder script per task — matching the Plans 01-01..01-04 convention so Phase 2 can clone the pattern.

## Key Changes

### notebook_kaggle.ipynb — Cells 9, 10, 11, 12, 13 (appended)

Notebook grew from 9 cells (Plan 01-04 ending state) to 14 cells (Plan 01-05 ending state). Each cell is a one-stage-one-unit addition.

**Cell 9 — Stage 4: RRF fuse.** Ports `rrf_fuse(rankings, k=60)` verbatim from `solution.py:220-226`. Loops over `_target_ids` (defined in Cell 7, already SMOKE-truncated for val) and for each qid fuses `[dense_laws_ids[qid], bm25_laws_ids[qid]]` with `k=RRF_K_CONST` (=60, from Cell 0). Populates `fused_laws_ids[qid]` as a list of `int` laws row indices sorted by fused score descending. Logs the top-5 fused indices for `_target_ids[0]` as a sanity sample.

**Cell 10 — Stage 5: Cross-encoder rerank + unload.** Imports `AutoModelForSequenceClassification` scoped to this cell (not top-of-notebook). Entire stage is wrapped in `if USE_RERANKER:` (INC-02). Loads `rerank_tok = AutoTokenizer.from_pretrained(str(RERANK_DIR))` and `rerank_mdl = AutoModelForSequenceClassification.from_pretrained(str(RERANK_DIR)).to(DEVICE).eval()`. Defines a local `rerank(query_en, candidate_texts, batch_size=32)` closure ported from `solution.py:194-214` — batches pairs through the cross-encoder under `torch.no_grad()`, returns a numpy score array. Per query: take `fused_laws_ids[qid][:RERANK_K]`, materialize candidate texts as `f"{citation} {text[:1500]}"` (1500-char cap to stay under the 512-token limit after tokenization), score with `rerank(q_en_canon[qid], cand_texts)`, and re-sort with `np.argsort(-scores)`. Populates `reranked_laws_ids[qid]`. After the loop: `unload(rerank_tok, rerank_mdl)` to free VRAM. The `USE_RERANKER=False` fallback branch sets `reranked_laws_ids[qid] = fused_laws_ids[qid][:RERANK_K]` so downstream cells always see a populated dict.

**Cell 11 — Stage 6a: Calibration.** Ports `macro_f1(gold_sets, pred_sets)` verbatim from `solution.py:288-300` — per-query F1 averaged, with the `(empty, empty) -> 1.0` and `(empty, non-empty) -> 0.0` edge cases intact. Ports `calibrate_top_k(val_results, k_min=1, k_max=60)` from `solution.py:303-313` with two modifications: `k_max=60` to match the plan-level hyperparameter (not solution.py's 80), and the CP-4 boundary WARN which fires if `best_k == k_min` or `best_k == k_max`. Builds `val_results` by iterating `val.iterrows()`, filtering to qids present in `reranked_laws_ids` (SMOKE mode may have dropped val rows), parsing `gold_citations` on `';'` into a set, and mapping `reranked_laws_ids[qid]` row indices through `canonicalize(laws.iloc[idx]['citation'])` to produce `ranked_citations`. Calls `best_k, best_f1 = calibrate_top_k(val_results, k_min=1, k_max=60)` — CALIB-05 enforced by function signature (takes only `val_results`). Prints the canonical `*** Val macro-F1 = X.XXXX  @  top-K ***` banner that matches the solution.py logging pattern.

**Cell 12 — Stage 6b: Validate + write.** Defines `validate_submission(df, test_df)` inline from the research §Submission Validator section. Asserts: columns == `["query_id", "predicted_citations"]`, `len(df) == len(test_df)`, `set(df["query_id"]) == set(test_df["query_id"])`, `df["query_id"].is_unique`, no NaN in `predicted_citations`, every row is `str`, no trailing `;`, and a soft WARN on comma-inside-citation (which is not an assertion because some Swiss citations legitimately contain commas in `BGE 141 III 188 E. 4.2` form — we only warn so the user can sanity-check). Iterates `test.iterrows()`, applies `best_k` to `reranked_laws_ids[qid]`, canonicalizes each resulting citation (CALIB-02), dedupes preserving order with a `seen` set, joins with `;`, and appends to `sub_rows`. Builds `submission_df = pd.DataFrame(sub_rows)[["query_id", "predicted_citations"]]` with explicit column ordering so validate_submission's column check is deterministic. Calls `validate_submission(submission_df, test)` (CALIB-03: validate BEFORE write). Writes `submission_df.to_csv(OUT_PATH, index=False)`. Logs path + row count.

**Cell 13 — Pipeline completion summary.** One-shot banner that prints `Mode: SMOKE|FULL`, `Val macro-F1`, `submission.csv rows`, `submission.csv path`, and a terminal hint pointing at the Task 4 human-verify checkpoint + `git tag phase-1-submission`. SMOKE gating is inline in Cells 6 and 7 (truncate laws and val), not Cell 13 — by the time control reaches Cell 13 the pipeline has already run end-to-end under whatever SMOKE flag was set in Cell 0. Cell 13 is a summary, not a runner.

### scripts/build_notebook_cells_9_10.py — new

Idempotent nbformat editor for Cells 9-10 (Task 1). Matches the Plans 01-01..01-04 builder convention: dependency-checks (Cell 7 has `dense_laws_ids`, Cell 8 has `bm25_laws_ids`, Cell 7 has `_target_ids`); appends or replaces in place; pops `id` from new cells to keep `nbformat_minor=4` readable; exits with distinct error codes on dependency failures.

### scripts/build_notebook_cells_11_12.py — new

Idempotent nbformat editor for Cells 11-12 (Task 2). Dependency-checks Cells 9-10 from Plan 01-05 Task 1 (Cell 10 has `reranked_laws_ids`, Cell 9 has `fused_laws_ids`).

### scripts/build_notebook_cell_13.py — new

Idempotent nbformat editor for Cell 13 (Task 3). Dependency-checks Cells 11-12 from Plan 01-05 Task 2.

## Commits

| Task | Hash      | Message                                                                                       |
| ---- | --------- | ---------------------------------------------------------------------------------------------- |
| 1    | `1ea37e2` | feat(01-05): add notebook Cells 9-10 (RRF fuse + cross-encoder rerank)                         |
| 2    | `bccd42b` | feat(01-05): add notebook Cells 11-12 (calibrate + validate + write submission.csv)            |
| 3    | `4b316e1` | feat(01-05): add notebook Cell 13 (pipeline completion summary) + 14-cell structural validation |
| 4    | —         | BLOCKED on human-verify checkpoint; user will manually create `phase-1-submission` git tag    |

## Verification Results

All Task 1-3 acceptance criteria pass on the final 14-cell notebook.

**Task 1 acceptance (Cells 9 and 10 present with required tokens):**

```
OK
```

Where `OK` expands to the five nbformat assertions from the plan:
- `len(nb.cells) >= 11` ✓
- Cell 9 contains `def rrf_fuse`, `RRF_K_CONST`, `fused_laws_ids` ✓
- Cell 10 contains `AutoModelForSequenceClassification`, `def rerank`, `reranked_laws_ids` ✓
- Cell 10 contains `USE_RERANKER`, `unload(rerank_tok, rerank_mdl)` ✓
- Cell 10 contains `q_en_canon[qid]` ✓

**Task 2 acceptance (Cells 11 and 12 present with required tokens):**

- `len(nb.cells) >= 13` ✓
- Cell 11 contains `def macro_f1`, `def calibrate_top_k`, `k_max=60`, `Val macro-F1` ✓
- Cell 11 contains `val_results`, `canonicalize(laws.iloc[idx]["citation"])` ✓
- Cell 12 contains `def validate_submission`, `submission_df.to_csv(OUT_PATH` ✓
- Cell 12 contains `reranked_laws_ids[qid][:best_k]`, `";".join(deduped)` ✓
- Cell 12 contains `validate_submission(submission_df, test)`, `assert not v.endswith(";")` ✓

**Task 3 acceptance (full 14-cell structural validation from plan action block):**

```
STRUCTURAL VALIDATION OK: 14 cells
```

Every cell 0-13 contains every token listed in the plan's `<action>` block check dictionary. Specifically verified:
- Cells 0-8 are byte-intact from Plans 01-01..01-04 (their markers all survive: `IS_KAGGLE`, `USE_FAISS_GPU`, `BGE_M3_DIR`, `OPUS_MT_DIR`, `USE_RERANKER`, `SMOKE_LAWS_N`, `assert torch.cuda.is_available()`, `DEVICE = torch.device("cuda")`, `def unload`, `pd.read_csv(DATA_DIR / "laws_de.csv")`, `def canonicalize`, `def extract_legal_codes`, `def tokenize_for_bm25_de`, `Non-collapsing canonicalize assert OK`, `MarianMTModel`, `def translate_en_de`, `translations`, `bm25_query_texts`, `q_en_canon`, `unload(opus_tok, opus_mdl)`, `AutoModel.from_pretrained(str(BGE_M3_DIR), torch_dtype=torch.float16)`, `last_hidden_state[:, 0]`, `torch.nn.functional.normalize(emb, p=2, dim=1)`, `_probe.shape == (2, 1024)`, `AH-1 TRIGGER`, `time.time() - t_start > 60`, `faiss.IndexFlatIP(1024)`, `if SMOKE:`, `def dense_query_embedding`, `faiss_index.search(q_vec, DENSE_LAWS_K)`, `dense_laws_ids[qid] = I[0].tolist()`, `unload(bge_tok, bge_mdl)`, `import bm25s`, `bm25s.BM25()`, `bm25_laws.index(laws_bm25_tokens)`, `bm25_laws.retrieve([q_tokens], k=BM25_LAWS_K)`, `bm25_laws_ids`).
- Cells 9-13 each contain every marker from the Task 3 check dict.
- Full notebook text contains NO `torch.device("cpu")` override anywhere (CUDA gate intact).
- Cell 5 does NOT reference `mean_pool` (BGE-M3 uses CLS-token pooling, not mean pooling).

Additional self-checks beyond the plan's acceptance list:
- Cells 9, 10, 11, 12, 13 source each parses as valid Python via `ast.parse()`.
- All three new builder scripts parse as valid Python via `ast.parse()`.
- `nbformat.read('notebook_kaggle.ipynb', 4)` succeeds on the final notebook.
- Commit messages contain no AI attribution per CLAUDE.md global rule.

**Runtime execution deferred to Kaggle.** No local copy of `laws_de.csv` / BGE-M3 weights / mmarco weights is present in the worktree, and the plan itself designates Task 4 (human-verify on Kaggle) as the single runtime validation gate for this plan. Cells 9-13 are therefore structurally verified here; end-to-end runtime behavior (including the actual `*** Val macro-F1 ***` value and `validate_submission OK: N rows` log) will be produced by the user's Kaggle run in Task 4.

## Deviations from Plan

### Scope clarifications (not Rule 1/2/3 auto-fixes)

- **Three builder scripts instead of inline nbformat edits.** The plan's action blocks could have been executed inline via `python -c` or a heredoc. The checked-in `scripts/build_notebook_cells_9_10.py`, `scripts/build_notebook_cells_11_12.py`, and `scripts/build_notebook_cell_13.py` match the convention from Plans 01-01..01-04 (each plan's builder script is committed alongside the notebook diff). This is tracked as a scope clarification, not an auto-fix, identical to the way Plans 01-02..01-04 called it out.
- **One throwaway validator script (`scripts/_validate_full_notebook_01_05.py`) was created during Task 3 to run the plan's structural check, then deleted before commit** — it's a one-shot runner, not a reusable artifact, and the plan never asks for it to be checked in. Its assertions are the source of the Task 3 `STRUCTURAL VALIDATION OK: 14 cells` line above. No commit for it, no trace in history.
- **Token markers kept strictly literal.** The plan's Task 3 structural check dict contains some entries with single-quoted strings and some with double-quoted (e.g. `'pd.read_csv(DATA_DIR / "laws_de.csv")'` vs `'def canonicalize'`). All builder scripts emit cell source that matches the plan's embedded Python exactly (including double-quoted dict keys like `canonicalize(laws.iloc[idx]["citation"])`), so every token in the check dict is present as a literal substring.
- **Dedupe inside `';'.join()` in Cell 12.** The plan's Task 2 action block includes a `seen = set(); deduped = []` loop before `';'.join(deduped)`. This is preserved verbatim — it's not a deviation, just explicitly flagged here because deduplication interacts with `canonicalize()` (CALIB-02). Because `canonicalize()` is applied BEFORE the `seen` set is consulted, two raw citations that differ only by case or whitespace collapse to a single canonical form and will dedupe correctly. This is the intended behavior.

### Auto-fixed Issues

None in Tasks 1-3. The plan's action blocks were followed verbatim; no bugs, no missing critical functionality, no blocking issues encountered. Task 4 is a human-verify gate so no auto-fixes are possible there by construction.

## Known Stubs

None introduced by this plan. Every cell 9-13 is fully wired to its upstream dependencies:
- Cell 9 reads `dense_laws_ids` (Cell 7) and `bm25_laws_ids` (Cell 8).
- Cell 10 reads `fused_laws_ids` (Cell 9), `q_en_canon` (Cell 4), `_target_ids` (Cell 7), `laws` (Cell 2/3), `DEVICE` (Cell 1), `RERANK_DIR` (Cell 0), `USE_RERANKER` (Cell 0), `RERANK_K` (Cell 0), and writes `reranked_laws_ids`.
- Cell 11 reads `reranked_laws_ids` (Cell 10), `val` (Cell 2), `laws` (Cell 2/3), `canonicalize` (Cell 3), and writes `val_results`, `best_k`, `best_f1`.
- Cell 12 reads `reranked_laws_ids` (Cell 10), `test` (Cell 2), `laws` (Cell 2/3), `canonicalize` (Cell 3), `best_k` (Cell 11), `OUT_PATH` (Cell 0), and writes `submission_df` + `submission.csv`.
- Cell 13 reads `SMOKE`, `OUT_PATH` (Cell 0), `best_f1`, `best_k` (Cell 11), `submission_df` (Cell 12). No further writes.

## Threat Flags

None beyond the register already documented in the plan's `<threat_model>`:
- **T-01-05-01 (Tampering on submission.csv format)** — mitigate. `validate_submission()` runs BEFORE `to_csv()` in Cell 12. CALIB-03 enforced.
- **T-01-05-02 (canonicalize drift)** — mitigate. Cell 3's non-collapsing assert fires at load time; Cell 12 re-applies `canonicalize()` on write so the write path is the single source of truth for the canonical form.
- **T-01-05-03 (calibrate boundary hit)** — mitigate. `calibrate_top_k` WARN on `best_k in {k_min, k_max}` fires as stderr in the Kaggle log; user sees it during Task 4 visual inspection.
- **T-01-05-04 (cross-encoder OOM)** — accept. `RERANK_K=150` * `batch_size=32` on T4 with cross-encoder in fp32 is well under the 16GB VRAM budget after the BGE-M3 unload.
- **T-01-05-05 (val-touches-test)** — mitigate. `calibrate_top_k(val_results)` signature takes only `val_results`; Cell 11 builds `val_results` from `val.iterrows()` only; Cell 12 applies `best_k` to `test.iterrows()` in a separate cell.
- **T-01-05-06 (git tag EoP)** — accept. Agent never runs `git tag` autonomously; user-gated at Task 4 checkpoint per CLAUDE.md.

No new trust boundaries, network endpoints, auth paths, or file-access patterns introduced.

## Deferred Issues

- **Pre-existing `nbformat_minor` mismatch** carried from Plans 01-01..01-04. The notebook declares `nbformat_minor=4` but some cells (Cells 0-3 from Plan 01-01) contain `id` fields that only exist in minor=5+. `nbformat.validate()` raises, but `nbformat.read(..., 4)` — which every plan acceptance check uses — works. Plan 01-05's three new builder scripts all pop `id` from freshly built cells (matching Plans 01-02..01-04). Deferred to a later cleanup pass; out of scope for Phase 1 per Rule 3 scope-boundary.
- **Runtime verification of Cells 9-13.** Deferred to the Kaggle human-verify run in Task 4. No local copy of `laws_de.csv` / BGE-M3 weights / mmarco weights is present in the worktree, and the plan itself delegates this to Task 4.
- **Wall-clock timing banner in Cell 13** is left as "(log manually via the cell timings above)" rather than wired to a global start-time variable, because no such variable exists at Cell 1 yet (Plans 01-01..01-04 never defined one). Filing as a deferred nicety — a `t_start = time.time()` at the top of Cell 1 plus a subtraction in Cell 13 would close this, but it's not in any plan.

## Files Touched

| File                                                              | Status   | Purpose                                                                                      |
| ----------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `notebook_kaggle.ipynb`                                           | modified | Cells 9-13 appended — RRF fuse, rerank, calibrate, validate, write, pipeline summary          |
| `scripts/build_notebook_cells_9_10.py`                            | created  | Idempotent nbformat editor for Cells 9-10 (Task 1)                                            |
| `scripts/build_notebook_cells_11_12.py`                           | created  | Idempotent nbformat editor for Cells 11-12 (Task 2)                                           |
| `scripts/build_notebook_cell_13.py`                               | created  | Idempotent nbformat editor for Cell 13 (Task 3)                                               |
| `.planning/phases/01-foundation-laws-pipeline/01-05-SUMMARY.md`   | created  | This summary (partial / checkpoint-pending)                                                   |

## Checkpoint: human-verify (Task 4 — PENDING)

Task 4 is a blocking `checkpoint:human-verify` gate. The agent has NOT executed it. The user must:

1. On Kaggle with T4 x2 accelerator enabled, upload BGE-M3, opus-mt-tc-big-en-de, and mmarco-reranker as Kaggle datasets per research §Model Download.
2. Open `notebook_kaggle.ipynb` in a fresh Kaggle kernel and set `SMOKE=1` env var for a first dry-run (completes in <5 minutes per FOUND-05), then unset and re-run for full inference.
3. Confirm the following log lines in order:
   - `Device: Tesla T4` / `VRAM total: ~15GB`
   - `Translated N queries in N.Ns`
   - `probe dim OK: (2, 1024)  std=...`
   - `60s checkpoint OK: N docs encoded, proceeding` (full run only)
   - `FAISS: N vectors, d=1024`
   - `Non-collapsing canonicalize assert OK on N unique citations`
   - `*** Val macro-F1 = X.XXXX  @  top-N ***`
   - `validate_submission OK: N rows`
   - `submission.csv written to: /kaggle/working/submission.csv  rows=N`
4. Inspect `submission.csv`:
   - Header row is `query_id,predicted_citations`.
   - Every test qid has a non-null row.
   - Citations are `;`-separated with no trailing `;`.
   - Sample 3 citations and confirm they match strings in `laws_de.csv` exactly.
5. Confirm val macro-F1 is meaningfully above the 0.009 baseline (any non-trivial positive F1 is a Phase 1 win; Phase 2 adds court coverage).
6. From repo root:
   ```bash
   git add notebook_kaggle.ipynb requirements.txt download_models.py .planning/REQUIREMENTS.md .planning/phases/01-foundation-laws-pipeline/
   git commit -m "phase 1: foundation + laws pipeline — val F1 X.XXXX"
   git tag phase-1-submission
   ```
   **Commit message must contain NO AI attribution per CLAUDE.md global rule.**

When the user replies with the val F1 value and confirmation that `git tag phase-1-submission` was created, a continuation agent should:
- Mark Task 4 done in this summary.
- Flip `status` frontmatter from `partial / checkpoint-pending` to `complete`.
- Update `tasks_completed: 3 -> 4` and remove the `tasks_pending` list.
- Add the user-reported val F1 value to the one-liner and metrics.
- Append the Task 4 commit hash to the Commits table.
- Append `INC-03` to `requirements_satisfied_structurally` (it will no longer be pending).
- Commit the updated summary.
- STATE.md and ROADMAP.md updates are intentionally deferred per this plan's execution contract — orchestrator handles those.

## Self-Check: PASSED (partial)

Verified all created files exist on disk in the worktree:

- `notebook_kaggle.ipynb` — FOUND (14 cells; Cells 9-13 contain every required marker from the Task 1-3 acceptance criteria and the full Task 3 structural validation dict)
- `scripts/build_notebook_cells_9_10.py` — FOUND (Python AST valid; idempotent; dependency-checks Cells 7-8)
- `scripts/build_notebook_cells_11_12.py` — FOUND (Python AST valid; idempotent; dependency-checks Cells 9-10)
- `scripts/build_notebook_cell_13.py` — FOUND (Python AST valid; idempotent; dependency-checks Cells 11-12)
- `.planning/phases/01-foundation-laws-pipeline/01-05-SUMMARY.md` — FOUND (this file)

Verified the three task commits exist on branch `worktree-agent-a20fbea9`:

- `1ea37e2` — FOUND (feat(01-05): add notebook Cells 9-10 (RRF fuse + cross-encoder rerank))
- `bccd42b` — FOUND (feat(01-05): add notebook Cells 11-12 (calibrate + validate + write submission.csv))
- `4b316e1` — FOUND (feat(01-05): add notebook Cell 13 (pipeline completion summary) + 14-cell structural validation)

Plan acceptance criteria — Tasks 1, 2, 3 all pass every listed check. Task 4 is BLOCKED on human-verify and is NOT claimed complete. This summary is therefore marked `status: partial / checkpoint-pending` in the frontmatter until the user confirms.
