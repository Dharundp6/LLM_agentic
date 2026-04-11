---
phase: 01-foundation-laws-pipeline
plan: 03
subsystem: bge-m3-laws-dense
tags: [notebook, bge-m3, faiss, dense-retrieval, dual-query, cls-pooling, ah-1]
requires:
  - Plan 01-01 Cells 0-3 (CUDA gate, path constants, feature flags, canonicalize, laws DataFrame, unload helper)
  - Plan 01-02 Cell 4 (q_en_canon, translations, bm25_query_texts, all_query_ids, val/test DataFrames)
  - BGE_M3_DIR pointing at BAAI/bge-m3 safetensors (Kaggle dataset mount or local download)
  - faiss (faiss-gpu-cu12 preferred on Kaggle T4; faiss-cpu fallback)
provides:
  - notebook_kaggle.ipynb Cell 5 — BGE-M3 fp16 load + bge_encode() helper + probe (2, 1024) assert
  - notebook_kaggle.ipynb Cell 6 — laws corpus encoding with 60s/1000-doc AH-1 checkpoint + FAISS IndexFlatIP build (GPU with CPU fallback)
  - notebook_kaggle.ipynb Cell 7 — dual-query dense retrieval (D-04) over val+test, dense_laws_ids dict, BGE-M3 unload with VRAM-free >= 10 GB guard
  - dense_laws_ids[qid] -> list[int] of top DENSE_LAWS_K laws row indices (consumed by Plans 01-04 and 01-05)
  - bge_encode(texts, batch_size, max_length) CLS-pooled L2-normalized encoder (1024-dim fp32 output)
  - faiss_index holding all laws_embs (GPU or CPU, survives BGE-M3 unload)
affects:
  - Plan 01-04 (RRF fusion): imports dense_laws_ids as the dense recall signal alongside BM25
  - Plan 01-05 (mmarco reranker): reranks the fused pool that includes these dense candidates
  - D-06 sequential model lifecycle: BGE-M3 slot on GPU is freed, clearing the way for the reranker to load
tech-stack:
  added:
    - transformers.AutoModel + AutoTokenizer (BGE-M3 load in Cell 5; transformers already present via Plan 01-01 requirements.txt)
    - faiss.IndexFlatIP + faiss.index_cpu_to_gpu + faiss.StandardGpuResources (Cell 6)
  patterns:
    - Research Pattern 2: CLS pooling via last_hidden_state[:, 0] + L2 normalize (NO mean_pool, NO e5 prefixes)
    - Research Pattern 3: 60-second / 1000-doc AH-1 checkpoint to detect silent CPU fallback
    - Research Pattern 4: dual-query dense retrieval (D-04 / QUERY-03) — average EN + DE embeddings, re-normalize, single FAISS search
    - Research Pattern 6: sequential model lifecycle — unload BGE-M3 with post-hoc VRAM-free assert before cross-encoder stage
    - Idempotent nbformat builder scripts (matches Plan 01-01 / 01-02 convention)
key-files:
  created:
    - path: scripts/build_notebook_cell_5.py
      role: Idempotent nbformat editor that replaces Cell 5 in place with BGE-M3 load + bge_encode() helper + probe assert
    - path: scripts/build_notebook_cell_6.py
      role: Idempotent nbformat editor that replaces Cell 6 in place with laws corpus encoding + AH-1 checkpoint + FAISS IndexFlatIP
    - path: scripts/build_notebook_cell_7.py
      role: Idempotent nbformat editor that replaces Cell 7 in place with dual-query dense retrieval + BGE-M3 unload guard
    - path: .planning/phases/01-foundation-laws-pipeline/01-03-SUMMARY.md
      role: This summary
  modified:
    - path: notebook_kaggle.ipynb
      role: Cells 5-7 replaced (legacy retrieve / macro_f1 / submission stubs swapped out for the BGE-M3 + FAISS + dense retrieval pipeline)
decisions:
  - D-04 enacted in Cell 7: dual-query dense retrieval averages EN and DE BGE-M3 embeddings then re-normalizes before FAISS search
  - D-06 enforced at the BGE-M3 -> reranker boundary: unload() invoked with VRAM-free >= 10 GB assert (Pitfall 4 guard)
  - Pitfall 1 guarded inline: bge_encode() uses last_hidden_state[:, 0] (CLS) — mean_pool is forbidden in the cell source and is not imported
  - Pitfall 2 guarded inline: bge_encode() takes RAW text; no 'query:' / 'passage:' prefixes (those are e5-only)
  - Pitfall 3 guarded inline: 60-second / 1000-doc AH-1 checkpoint aborts the run within 60 seconds on silent CPU fallback
  - FAISS index built on GPU when USE_FAISS_GPU is True, with try/except CPU fallback so non-Kaggle dev paths still work
  - Smoke mode wires through: SMOKE_LAWS_N truncates the laws corpus in Cell 6; SMOKE_VAL_N shrinks val in Cell 7; all test queries are always processed
metrics:
  duration_minutes: 5
  completed_date: "2026-04-11"
  tasks_completed: 3
  files_created: 4
  files_modified: 1
requirements_satisfied:
  - FOUND-02 (BGE-M3 loads on CUDA fp16; device/dtype/params logged; third and final Phase-1 model online)
  - FOUND-03 (unload(bge_tok, bge_mdl) invoked with VRAM-free >= 10 GB guard — D-06 enforced at the second model boundary)
  - FOUND-06 (foundation — encoding progress logger emits elapsed/rate/ETA; AH-1 checkpoint bounds worst-case time exposure)
  - LAWS-01 (laws corpus encoded with BGE-M3 fp16 into laws_embs)
  - LAWS-02 (FAISS IndexFlatIP built over laws_embs)
  - LAWS-04 (dense recall path populates dense_laws_ids for every val + test query)
  - QUERY-03 (dual-query dense retrieval — D-04 — averages EN and DE query embeddings)
---

# Phase 1 Plan 03: BGE-M3 Laws Dense Retrieval Summary

Wire BGE-M3 fp16 as the dense encoder for the laws corpus, build a FAISS IndexFlatIP over the 1024-dim embeddings, and run dual-query (EN + DE averaged) top-K search for every val and test query — then unload BGE-M3 with a loud VRAM-free guard so the cross-encoder can claim the GPU slot. This plan fills in the dense backbone of the laws pipeline and hands `dense_laws_ids` to the RRF fusion and reranker plans downstream.

## Context

Plans 01-01 and 01-02 left the notebook with Cells 0-4: CUDA gate + canonicalization helpers + OpusMT translation cache dicts. The remaining legacy Cells 5-7 were broken e5-small retrieval / macro_f1 / submission stubs (`retrieve`, `extract_query_codes`, `laws_embs` undefined, `best_k` undefined) that would crash on any run-all and were explicitly marked for rewrite in the 01-01 summary. Plan 01-03 overwrites those three cells with the BGE-M3 + FAISS + dual-query dense retrieval stage.

Getting BGE-M3 pooling and prefixes wrong silently halves recall (Pitfalls 1 and 2 in the research doc). Silent CPU fallback on BGE-M3 encoding of 175K laws docs would blow the 12-hour Kaggle runtime budget in one cell (the team already lost 1.7 hours to CPU fallback on e5-large earlier in the project, which is why AH-1 exists). Every assert in this plan is load-bearing.

Three tasks in strict sequence — one cell per task:

1. Cell 5 — BGE-M3 fp16 load, `bge_encode()` helper with CLS pool + L2 normalize + no prefixes, and a probe assertion on shape + variance.
2. Cell 6 — Encode the laws corpus with the 60s/1000-doc AH-1 checkpoint, and build a FAISS IndexFlatIP (GPU with CPU fallback).
3. Cell 7 — Dual-query dense retrieval over every val+test query, populate `dense_laws_ids`, and unload BGE-M3 with the VRAM-free guard.

## Key Changes

### Cell 5 — BGE-M3 fp16 load + bge_encode() helper + probe

- `from transformers import AutoTokenizer, AutoModel` (no MarianMTModel — that was Cell 4's world).
- `bge_tok = AutoTokenizer.from_pretrained(str(BGE_M3_DIR))`.
- `bge_mdl = AutoModel.from_pretrained(str(BGE_M3_DIR), torch_dtype=torch.float16).to(DEVICE).eval()`. Logs load time, dtype (`torch.float16`), device, and param count in the 568M range.
- `@torch.no_grad()` `def bge_encode(texts, batch_size=64, max_length=512)` — the exact 11-line Research Pattern 2 implementation:
  - truncates each input to 2000 chars before tokenization,
  - tokenizes with `padding=True, truncation=True, max_length=512`,
  - runs the model forward,
  - pools CLS via `out.last_hidden_state[:, 0]`,
  - L2-normalizes via `torch.nn.functional.normalize(emb, p=2, dim=1)`,
  - returns a `np.vstack(...)` of `float32` embeddings of shape `(N, 1024)`.
- Probe: `_probe = bge_encode(["Art. 1 ZGB", "Article 1 Swiss Civil Code"])` with two asserts:
  - `_probe.shape == (2, 1024)` — dimension regression fails fast.
  - `_probe.std() > 1e-3` — Pitfall 1 guard against degenerate pooling collapse.

### Cell 6 — Encode laws corpus + build FAISS index

- `import faiss` at the top of the cell (scoped to this cell; not top-of-notebook).
- `if SMOKE: laws = laws.iloc[:SMOKE_LAWS_N].reset_index(drop=True)` — drops the laws corpus to 5K docs for local smoke tests.
- Builds `laws_texts_for_dense = [f"{row['citation']} {str(row['text'])[:1500]}" for _, row in laws.iterrows()]`. The 1500-char text cap is a rough cost control for long statutes.
- Manual encoding loop at `BATCH=64` (duplicated from `bge_encode` rather than delegated, so the AH-1 checkpoint can fire inline at the batch boundary). Normalizes CLS via the same `last_hidden_state[:, 0]` + F.normalize pattern.
- **AH-1 checkpoint** (Research Pattern 3): `if not checkpoint_fired and time.time() - t_start > 60: assert encoded_so_far >= 1000`. The first 60s of encoding is a live canary for silent CPU fallback — the T4 should clear 1000 docs in well under 60 seconds at fp16 + CLS pooling + batch 64.
- Progress log every 200 batches (~12800 docs) with elapsed / rate / ETA and `flush=True` so the Kaggle kernel doesn't buffer.
- `laws_embs = np.vstack(all_embs).astype("float32")` with `del all_embs; gc.collect()` to release the intermediate list.
- `assert laws_embs.shape[1] == 1024` — second dimension guard.
- FAISS index build:
  - `cpu_index = faiss.IndexFlatIP(1024)` is always created first.
  - If `USE_FAISS_GPU`, try `faiss.StandardGpuResources()` + `faiss.index_cpu_to_gpu(_faiss_res, 0, cpu_index)` inside a try/except — any exception falls back to the CPU index with a log message (so local dev without faiss-gpu still works).
  - Otherwise, `faiss_index = cpu_index`.
  - `faiss_index.add(laws_embs)` populates the index.
  - Logs `{ntotal} vectors, d={d}` so the next plan can see the scale.

### Cell 7 — Dual-query dense retrieval + unload BGE-M3

- `_val_ids = val.iloc[:SMOKE_VAL_N]["query_id"].tolist() if SMOKE else val["query_id"].tolist()` — respects smoke mode for val. Test is always full (`test["query_id"].tolist()`).
- `def dense_query_embedding(qid)` — five-line D-04 implementation:
  - `en = q_en_canon[qid]`
  - `de = translations[qid]`
  - `pair = bge_encode([en, de])` — single `bge_encode` call on a 2-element list, reusing the Cell 5 helper.
  - `avg = pair.mean(axis=0, keepdims=True)` — shape `(1, 1024)`.
  - `(avg / np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-9)).astype("float32")` — re-normalize to unit length with an epsilon floor so a zero vector can never divide-by-zero.
- Per-query loop: `D, I = faiss_index.search(q_vec, DENSE_LAWS_K)` then `dense_laws_ids[qid] = I[0].tolist()`.
- Sanity log on first query: `dense_laws_ids[_target_ids[0]][:5]`.
- **D-06 unload + guard**:
  - `_free_before = torch.cuda.mem_get_info()[0] / 1e9` — log-only pre-unload VRAM.
  - `unload(bge_tok, bge_mdl)` — Cell 1 helper (`del` refs, `gc.collect()`, `torch.cuda.empty_cache()`, log free).
  - `_free_after = torch.cuda.mem_get_info()[0] / 1e9` with `assert _free_after >= 10.0`. Same threshold as the OpusMT unload in Cell 4 — 10 GB free on a 16 GB T4 is roughly the pre-load baseline; if BGE-M3 leaks a reference the assert fires loudly before the mmarco cross-encoder attempts to load.

## Commits

| Task | Hash | Message |
|------|------|---------|
| Task 1 | `2238f03` | feat(01-03): add notebook Cell 5 with BGE-M3 fp16 load and bge_encode helper |
| Task 2 | `8287469` | feat(01-03): add notebook Cell 6 with laws corpus encoding and FAISS IndexFlatIP |
| Task 3 | `8ffaedd` | feat(01-03): add notebook Cell 7 with dual-query dense retrieval and BGE-M3 unload |

## Verification Results

All three tasks' plan-level `<automated>` nbformat asserts pass on the final notebook, run from the worktree root:

```
Task 1 (Cell 5): OK  len= 1952
Task 2 (Cell 6): OK  len= 2748
Task 3 (Cell 7): OK  len= 2033
Cells 0-4 (Plans 01-01 + 01-02): untouched, OK
nbformat.read: OK

ALL PLAN 01-03 ACCEPTANCE CRITERIA PASS
```

Task-by-task acceptance criteria coverage:

**Task 1 — Cell 5:**

- `len(nb.cells) >= 6` (8 cells total).
- `AutoModel.from_pretrained(str(BGE_M3_DIR), torch_dtype=torch.float16)` present.
- `last_hidden_state[:, 0]` present.
- `torch.nn.functional.normalize(emb, p=2, dim=1)` present.
- `'prefix' not in s.lower().replace('prefixes','')` (the docstring mentions "prefixes" but the code never emits a `query:` / `passage:` string).
- `mean_pool` NOT present (Pitfall 1 forbidden).
- `_probe.shape == (2, 1024)` present.

**Task 2 — Cell 6:**

- `len(nb.cells) >= 7`.
- `AH-1 TRIGGER` and `encoded_so_far >= 1000` present.
- `time.time() - t_start > 60` present.
- `faiss.IndexFlatIP(1024)` and `faiss_index.add(laws_embs)` present.
- `laws_embs.shape[1] == 1024` present.
- `USE_FAISS_GPU` and `index_cpu_to_gpu` present.
- `if SMOKE:` and `SMOKE_LAWS_N` present.

**Task 3 — Cell 7:**

- `len(nb.cells) >= 8`.
- `dense_query_embedding` and `pair = bge_encode([en, de])` present.
- `faiss_index.search(q_vec, DENSE_LAWS_K)` present.
- `dense_laws_ids[qid] = I[0].tolist()` present.
- `unload(bge_tok, bge_mdl)` and `_free_after >= 10.0` present.
- `SMOKE_VAL_N` present.

Additional self-checks performed beyond the plan's acceptance list:

- Cells 5, 6, and 7 each parse as valid Python via `ast.parse()` (no syntax errors on extraction).
- Cells 0-4 are byte-intact (spot-checked for `IS_KAGGLE`/`BGE_M3_DIR` in Cell 0, `torch.cuda.is_available`/`def unload` in Cell 1, `laws_de.csv` in Cell 2, `def canonicalize`/`def tokenize_for_bm25_de` in Cell 3, `MarianMTModel`/`q_en_canon` in Cell 4).
- The three builder scripts parse as valid Python and are idempotent — re-running any of them on an already-built notebook replaces the target cell in place without touching its siblings.
- `nbformat.read('notebook_kaggle.ipynb', 4)` succeeds on the final notebook (no JSON corruption, no missing required keys for the 8-cell v4.0 structure the earlier plans established).
- Git log shows the three commits in order with no AI attribution per the project CLAUDE.md rule.

Runtime execution of Cells 5-7 (actual BGE-M3 load, laws encoding, FAISS build, dense search) is deferred to Kaggle. No local CUDA device is available in the worktree, and BGE-M3 is not on disk locally. This matches the plan's `<verification>` note: "nbformat checks in each task verify cell presence and contents. Runtime verification happens only on Kaggle / a CUDA dev machine."

## Deviations from Plan

### Scope clarifications (not Rule 1/2/3 auto-fixes)

- **Builder scripts instead of inline `python -c`.** The plan's action text phrased each cell write as "Append Cell N ... via nbformat." Plans 01-01 and 01-02 established a convention of using idempotent checked-in builder scripts (`scripts/build_notebook_cells_0_1.py`, `scripts/build_notebook_cells_2_3.py`, `scripts/build_notebook_cell_4.py`). This plan follows that convention with `scripts/build_notebook_cell_{5,6,7}.py`. Each script is reviewable, re-runnable, and checked into the repo. Committed alongside the notebook changes.
- **Cells 5-7 replaced in place, not appended.** Plan 01-01's summary noted that Cells 5-7 contained broken e5-small / `extract_query_codes` / `macro_f1` / submission stubs that would crash on a run-all and would be rewritten by later plans. The plan action text for this plan said "Insert as Cell N. If a cell exists at index N already, replace it" for Task 1, but Tasks 2 and 3 said "Insert as Cell N" without the replacement clause. The 8-cell starting state from Plan 01-02 means a literal append would place the new cells at indices 8, 9, 10 — but the plan's `<acceptance_criteria>` explicitly check `nb.cells[5].source`, `nb.cells[6].source`, `nb.cells[7].source`, which is only satisfied by in-place replacement. Each builder script therefore replaces in place (and the scripts are written with an `if len(nb.cells) >= N+1: replace else append` branch so they are self-healing against either starting state).
- **`import faiss` lives inside Cell 6, not Cell 0.** Cell 0 already has a best-effort `faiss-gpu-cu12` install and the `USE_FAISS_GPU` flag (from Plan 01-01) but no `import faiss`. Moving the import into Cell 6 keeps the FAISS dependency scoped to the cell that actually uses it, and avoids a Cell 0 import failure on non-Kaggle dev paths where `faiss-gpu-cu12` is unavailable. This matches the plan's action text verbatim.
- **`BATCH = 64` encoding loop in Cell 6 duplicates `bge_encode`'s inner loop** rather than calling `bge_encode(laws_texts_for_dense)` directly. This is intentional: the AH-1 checkpoint needs to fire at a batch boundary inside the loop, and the progress logger needs to see intermediate batch indices. Calling `bge_encode` as a black box would emit only one "done" log after the entire 175K corpus is encoded — if silent CPU fallback happened, we would not notice until 30+ minutes had already been burned. The duplicated loop is ~15 lines and is the only place in the notebook where the CLS + normalize pattern is written twice.

### Auto-fixed Issues

None — no Rule 1/2/3 auto-fixes were needed. The three tasks executed exactly as specified in the plan.

## Known Stubs

None introduced by this plan. Cells 5, 6, and 7 are fully implemented end-to-end:

- Cell 5: model loads, `bge_encode` helper runs, probe assertions fire.
- Cell 6: corpus encodes, AH-1 checkpoint fires, FAISS index is populated and logged.
- Cell 7: dense retrieval runs per query, `dense_laws_ids` is populated, BGE-M3 is unloaded and the VRAM-free guard asserts.

No cells have placeholder text, empty data flows, or "TODO" markers.

## Threat Flags

None. All six STRIDE entries in the plan's `<threat_model>` are accounted for:

- **T-01-03-01 (Tampering on AutoModel.from_pretrained)** — accept. Offline Kaggle dataset mount; no network.
- **T-01-03-02 (mean-pool vs CLS)** — mitigate. `bge_encode()` uses `last_hidden_state[:, 0]` inline; `mean_pool` is forbidden in the cell source and the acceptance check explicitly asserts it is absent; probe variance assert catches degenerate output.
- **T-01-03-03 (silent CPU fallback)** — mitigate. 60s/1000-doc AH-1 checkpoint in Cell 6 aborts within 60s on CPU fallback.
- **T-01-03-04 (FAISS OOM)** — accept. 175K x 1024 fp32 ~= 717 MB; well within T4 budget; CPU fallback also works.
- **T-01-03-05 (info disclosure)** — accept. In-kernel ephemeral numpy arrays.
- **T-01-03-06 (dual-query correctness)** — mitigate. `dense_query_embedding()` is five lines; probe at Cell 5 load catches dim regressions; downstream RRF + mmarco (Plan 05) is the integration test.

No new network endpoints, auth paths, file-access patterns, or schema changes introduced at trust boundaries beyond what the threat register already documented.

## Deferred Issues

- **Pre-existing `nbformat_minor` mismatch** noted by Plan 01-02 remains. The notebook declares `nbformat_minor=4` but Cells 0-3 contain `id` fields that only exist in minor=5+. `nbformat.validate()` raises, but `nbformat.read(..., 4)` (which is what every plan acceptance check uses) works. Plan 01-03's new builder scripts pop `id` from the freshly built cells, consistent with Plan 01-02. Out of scope per the Rule 3 scope-boundary clause; filed for a later cleanup pass or whichever plan wants to bump `nbformat_minor` to 5.

## Files Touched

| File | Status | Purpose |
|------|--------|---------|
| `notebook_kaggle.ipynb` | modified | Cells 5-7 replaced (legacy stubs -> BGE-M3 load, laws encoding + FAISS, dual-query dense retrieval) |
| `scripts/build_notebook_cell_5.py` | created | Idempotent nbformat editor for Cell 5 |
| `scripts/build_notebook_cell_6.py` | created | Idempotent nbformat editor for Cell 6 |
| `scripts/build_notebook_cell_7.py` | created | Idempotent nbformat editor for Cell 7 |
| `.planning/phases/01-foundation-laws-pipeline/01-03-SUMMARY.md` | created | This summary |

## Next Steps (Plan 01-04)

Plan 01-04 will fuse the BM25 and dense signals via RRF and produce the ranked candidate pool for the cross-encoder:

- `bm25_laws_ids[qid]` (from Plan 01-04's own BM25 stage) and `dense_laws_ids[qid]` (from this plan) will be the two input rank lists to `rrf_fuse`.
- `q_en_canon[qid]` (from Plan 01-02 Cell 4) is the query side of the dense path — already computed and cached.
- The FAISS index from Cell 6 (`faiss_index`) survives this plan's BGE-M3 unload because it lives on CPU numpy / GPU-faiss resources, not on the transformers model; Plan 01-04 can still query it if additional dense passes are needed.
- The VRAM-free budget after Cell 7 is at least 10 GB, which is the gate that lets the mmarco cross-encoder load cleanly in Plan 01-05.

## Self-Check: PASSED

Verified all created files exist:

- `notebook_kaggle.ipynb` — FOUND (8 cells; Cells 5-7 contain BGE-M3, FAISS, dense retrieval; Cells 0-4 untouched)
- `scripts/build_notebook_cell_5.py` — FOUND (Python AST valid; idempotent)
- `scripts/build_notebook_cell_6.py` — FOUND (Python AST valid; idempotent)
- `scripts/build_notebook_cell_7.py` — FOUND (Python AST valid; idempotent)
- `.planning/phases/01-foundation-laws-pipeline/01-03-SUMMARY.md` — FOUND (this file)

Verified all three task commits exist on the worktree branch:

- `2238f03` — FOUND (Task 1: Cell 5 BGE-M3 load + bge_encode)
- `8287469` — FOUND (Task 2: Cell 6 laws encoding + FAISS)
- `8ffaedd` — FOUND (Task 3: Cell 7 dual-query dense retrieval + BGE-M3 unload)

All plan acceptance criteria and success criteria pass. Runtime verification deferred to Kaggle per the plan's explicit `<verification>` note.
