---
phase: 01-foundation-laws-pipeline
plan: 01
subsystem: notebook-scaffold
tags: [notebook, scaffold, cuda, canonicalization, kaggle-paths, bm25s, charsplit, nltk]
requires:
  - Kaggle T4 x2 GPU accelerator
  - BAAI/bge-m3 + Helsinki-NLP/opus-mt-tc-big-en-de packaged as Kaggle datasets
  - laws_de.csv / val.csv / test.csv in competition dataset mount
provides:
  - notebook_kaggle.ipynb Cells 0-3 wired top-to-bottom
  - IS_KAGGLE / DATA_DIR / BGE_M3_DIR / OPUS_MT_DIR / RERANK_DIR / OUT_PATH path switch
  - USE_RERANKER / USE_COURT_CORPUS / USE_ENTITY_GRAPH / USE_COURT_DENSE_RERANK / USE_JINA_CROSS_ENCODER / USE_LLM_AUGMENTATION / SMOKE feature flags
  - BM25_LAWS_K / DENSE_LAWS_K / RRF_K_CONST / RERANK_K / SMOKE_LAWS_N / SMOKE_VAL_N / SEED constants
  - unload() GPU memory helper (D-06 sequential model lifecycle)
  - canonicalize() single source of truth for Swiss citation format
  - extract_legal_codes() regex extractor for QUERY_CODES ∪ BGE references
  - tokenize_for_bm25_de() legal-aware German tokenizer with optional CharSplit decompounding
  - LAW_CODE_ALIASES French->German map (CC->ZGB, CO->OR, CP->StGB, CPP->StPO, LP->SchKG, LTF->BGG)
  - GERMAN_STOP NLTK stopword frozenset
  - Laws corpus canonicalized in place (LAWS-05) with non-collapsing guard
affects:
  - Plan 01-02 onward: can import these helpers and path constants without redefinition
  - REQUIREMENTS.md traceability: QUERY-01 moved from Phase 2 to Phase 1
tech-stack:
  added:
    - bm25s>=0.2.0 (requirements.txt + Kaggle runtime install)
    - CharSplit>=0.5 (requirements.txt + Kaggle runtime install)
    - nltk>=3.8 (requirements.txt + Kaggle runtime install)
  patterns:
    - IS_KAGGLE branching path switch (extended from solution.py:20-35)
    - Hard-fail CUDA gate at Cell 1 (FOUND-01 / AH-1 prevention)
    - Sequential model lifecycle via unload() helper
    - Single-source-of-truth canonicalization pipeline (query + corpus + submission)
    - Non-collapsing canonicalize guard (Pitfall 6)
key-files:
  created:
    - path: notebook_kaggle.ipynb
      role: Cells 0-3 rebuilt (pip installs, CUDA assert, CSV load, canonicalization helpers); Cells 4-7 remain legacy and will be rewritten by Plan 01-02+
    - path: requirements.txt
      role: Added bm25s / CharSplit / nltk pins for local dev
    - path: download_models.py
      role: Added BGE-M3 and opus-mt-tc-big-en-de to MODELS map
    - path: scripts/build_notebook_cells_0_1.py
      role: Idempotent nbformat editor for Cells 0 and 1 (Task 2)
    - path: scripts/build_notebook_cells_2_3.py
      role: Idempotent nbformat editor for Cells 2 and 3 (Task 3)
  modified:
    - path: .planning/REQUIREMENTS.md
      role: QUERY-01 traceability row moved Phase 2 -> Phase 1 (enacts D-01)
decisions:
  - D-01 enacted in REQUIREMENTS.md: OpusMT translation pulled forward into Phase 1
  - D-05 enacted in Cell 3: canonicalize() is the single source of truth for query, corpus, and submission citation formatting
  - D-06 foundation laid in Cell 1: unload() helper present; sequential model lifecycle is enforceable from Plan 01-02 onward
  - Legacy notebook cells 4-7 left in place (out of this plan's scope; Plan 01-02+ will rebuild retrieval and evaluation cells)
metrics:
  duration_minutes: 6
  completed_date: "2026-04-11"
  tasks_completed: 3
  files_created: 5
  files_modified: 1
requirements_satisfied:
  - FOUND-01
  - FOUND-02 (partial — device logged; model loads come in Plan 01-02+)
  - FOUND-03 (foundation — unload() helper present)
  - FOUND-04
  - FOUND-06 (foundation — hyperparams pinned; time-budget tracked from Plan 01-02+)
  - LAWS-05
  - QUERY-01 (traceability only — translation model wiring comes in Plan 01-02+)
  - QUERY-04
  - INC-02
---

# Phase 1 Plan 01: Notebook Scaffold + CUDA Hard-Assert + Canonicalization Helpers Summary

Rebuild the Kaggle notebook's first four cells so that Phase 1 can never silently run on CPU, paths/feature-flags/hyperparams are defined once at the top, and a single-source-of-truth `canonicalize()` helper governs every citation string that flows through query parsing, corpus indexing, and submission writing.

## Context

The submitted notebook was scoring val macro-F1 = 0.009 because it force-disabled CUDA at the top of the file (`DEVICE = torch.device('cpu')`) and used e5-small with no canonicalization. This plan rips out that force-CPU override, replaces it with a hard-fail CUDA gate (FOUND-01 / AH-1 prevention), and lays down the canonicalize / extract_legal_codes / tokenize_for_bm25_de helpers plus the LAWS-05 in-place canonicalization of the laws corpus that every subsequent retrieval plan depends on.

Three tasks in strict sequence:

1. Update `requirements.txt`, `download_models.py`, and `.planning/REQUIREMENTS.md` traceability.
2. Rewrite notebook Cells 0 and 1 (pip installs, path constants, feature flags, hyperparams, CUDA assert, `unload()` helper).
3. Add notebook Cells 2 and 3 (CSV load + canonicalization helpers + LAWS-05 apply + non-collapsing guard).

## Key Changes

### Cell 0 — bootstrap
- Kaggle-only `pip install -q bm25s>=0.2.0 CharSplit nltk` with best-effort `faiss-gpu-cu12>=1.14` and `USE_FAISS_GPU` fallback flag.
- Downloads NLTK german stopwords (quiet) on Kaggle.
- Extends solution.py:20-35 `IS_KAGGLE` path switch with `BGE_M3_DIR` and `OPUS_MT_DIR`.
- Defines the full INC-02 feature-flag block: `USE_RERANKER`, `USE_COURT_CORPUS`, `USE_ENTITY_GRAPH`, `USE_COURT_DENSE_RERANK`, `USE_JINA_CROSS_ENCODER`, `USE_LLM_AUGMENTATION`, plus `SMOKE` from the `SMOKE` env var.
- Hyperparameter block: `BM25_LAWS_K=100`, `DENSE_LAWS_K=100`, `RRF_K_CONST=60`, `RERANK_K=150`, `SMOKE_LAWS_N=5000`, `SMOKE_VAL_N=3`, `SEED=42`.

### Cell 1 — hard gate
- `assert torch.cuda.is_available()` with AH-1 abort message — this is the reason Phase 1 exists.
- Logs `torch.cuda.get_device_name(0)` and total/free VRAM via `torch.cuda.mem_get_info()`.
- Seeds `random`, `np.random`, `torch`, and `torch.cuda` from `SEED`.
- `unload(*objs)` helper: del refs, gc collect, `torch.cuda.empty_cache()`, log VRAM free — reusable by every model stage in Plan 01-02+ for D-06.

### Cell 2 — data load
- Asserts `DATA_DIR.exists()` before reading.
- Loads `laws_de.csv`, `val.csv`, `test.csv` with `.fillna("")`.
- Logs shape, columns, and `memory_usage(deep=True)` for laws; shapes for val/test.
- Prints A1-A10 corpus inspection banner (unique citation count, sample of 5 citations, query counts).

### Cell 3 — canonicalization
- `LAW_CODE_ALIASES`: CC→ZGB, CO→OR, CP→StGB, CPP→StPO, LP→SchKG, LTF→BGG.
- `canonicalize(s)`: strip + collapse whitespace, `\u00df→ss`, normalize `Art./Abs./lit./Ziff.` punctuation, apply aliases. Idempotent; inline-asserted with `canonicalize("Art.11 Abs.2 CC") == "Art. 11 Abs. 2 ZGB"` and a round-trip check.
- `extract_legal_codes(text)`: QUERY_CODES regex over {ZGB, OR, StGB, StPO, BGG, SchKG, BGB, AHVG, IVG, ELG, KVG, UVG, BVG, DBG, MWSTG, USG, RPG}; also captures `Art. N CODE` and `BGE N [Roman] N` patterns.
- `tokenize_for_bm25_de(text)`: lowercase, `re.findall(r"[\w.]+")`, filter GERMAN_STOP from NLTK, optionally decompound tokens > 3 chars via CharSplit (append head/tail on score > 0.5, fail-soft if library missing). Inline-asserted to preserve `art.` and `or` tokens.
- LAWS-05 apply: `laws["citation"] = laws["citation"].apply(canonicalize)` and same for `laws["text"]`.
- Pitfall 6 guard: non-collapsing assert over `laws["citation"].drop_duplicates()` aborting if any distinct pair collapses.

## Commits

| Task | Hash | Message |
|------|------|---------|
| Task 1 | `a6aaa38` | chore(01-01): add bm25s/CharSplit/nltk + BGE-M3/opus-mt-tc-big download mapping |
| Task 2 | `8db77d8` | feat(01-01): rewrite notebook Cells 0-1 with CUDA hard-assert + path constants |
| Task 3 | `570a0bf` | feat(01-01): add notebook Cells 2-3 with CSV load + canonicalization helpers |

## Verification Results

Plan-level verification (`python scripts`-driven nbformat asserts) — all pass:

- Cell 0 contains `IS_KAGGLE`, `USE_FAISS_GPU`, `BGE_M3_DIR`, `OPUS_MT_DIR`, `USE_RERANKER`, `USE_LLM_AUGMENTATION`, `SEED`, `BM25_LAWS_K`, `RERANK_K`.
- Cell 1 contains `assert torch.cuda.is_available()`, `DEVICE = torch.device("cuda")`, `get_device_name`, `mem_get_info`, `def unload`.
- Cell 2 contains `laws_de.csv`, `val.csv`, `test.csv`, `laws.shape` log.
- Cell 3 contains `def canonicalize`, `def extract_legal_codes`, `def tokenize_for_bm25_de`, `LAW_CODE_ALIASES` with `"CC":  "ZGB"`, `GERMAN_STOP`, `CharSplit`, `Non-collapsing canonicalize assert OK`, `laws["citation"] = laws["citation"].apply(canonicalize)`.
- Idempotence example `Art. 11 Abs. 2 ZGB` present in notebook JSON.
- Standalone Python smoke test of `canonicalize()` logic passes on 8 hand-picked inputs (including sharp-s round-trip, idempotence, French→German, whitespace normalization, BGE preservation).
- No `torch.device("cpu")` string present anywhere in the notebook — force-CPU override fully removed.
- `python -c "import ast; ast.parse(open('download_models.py').read())"` passes (syntax valid).
- `requirements.txt` contains `bm25s>=0.2.0`, `CharSplit>=0.5`, `nltk>=3.8`.
- `download_models.py` contains `BAAI/bge-m3` and `Helsinki-NLP/opus-mt-tc-big-en-de`.
- `.planning/REQUIREMENTS.md` shows `| QUERY-01 | Phase 1 | Pending |` (moved from Phase 2).

Runtime execution of the CUDA assert and LAWS-05 canonicalize is deferred to Kaggle (no local GPU; no laws_de.csv in repo). This is expected per the plan's Verification note.

## Deviations from Plan

### Scope clarifications (not rule-1/2/3 auto-fixes)

- **Worktree file bootstrap.** The sparse worktree only tracked `.planning/` and `CLAUDE.md`. `notebook_kaggle.ipynb`, `requirements.txt`, `download_models.py`, and `solution.py` existed in the main repo but were untracked by git. Copied the four files into the worktree at the start of execution so Task 1 and Tasks 2-3 could edit them. `solution.py` was read-only reference material and is NOT committed (left as untracked; it is not listed in the plan's `files_modified`).
- **Legacy cells 4-7 left untouched.** The original notebook had 8 cells (markdown header + 7 code cells). Tasks 2 and 3 replace Cells 0-3, which shifts the former Cells 2-7 to indices 4-9 but overwrites the old markdown header and force-CPU cell. Cells 4-7 reference symbols (`laws_cits`, `laws_embs`, `TOP_TRAIN`, `CODE_PAT`, `extract_query_codes`, `best_k`) that Plan 01-01 does not produce — they would crash on a run-all. Per SCOPE BOUNDARY and the plan's explicit instruction that Cells 4+ belong to Plan 01-02+, these cells are left in place untouched. Plan 01-02 (OpusMT translation) and later plans will rewrite them.
- **Helper script directory.** Created `scripts/` directory in the worktree and added `scripts/build_notebook_cells_0_1.py` + `scripts/build_notebook_cells_2_3.py`. The plan suggested nbformat edits could be done inline via `python -c`, but those two one-off scripts are easier to review, idempotent, and checked into the repo for reproducibility. Both committed with their respective tasks.

### Auto-fixed Issues

None — no Rule 1/2/3 auto-fixes were needed. The three tasks executed exactly as written.

## Known Stubs

None. All helpers in Cell 3 are fully implemented. The legacy Cells 4-7 are stubs from the user's perspective of "Phase 1 laws pipeline" but they are explicitly out of this plan's scope and tracked as Plan 01-02+ work.

## Threat Flags

None. No new network endpoints, auth paths, file-access patterns, or schema changes introduced at trust boundaries beyond what the threat register in 01-01-PLAN.md already documented. The mitigations for T-01-01-03 (CUDA hard-assert) and T-01-01-05 (LAWS-05 + non-collapsing guard) are both implemented as planned.

## Deferred Issues

None.

## Files Touched

| File | Status | Purpose |
|------|--------|---------|
| `requirements.txt` | created (new in branch) | bm25s + CharSplit + nltk pins |
| `download_models.py` | created (new in branch) | BGE-M3 + opus-mt-tc-big entries |
| `notebook_kaggle.ipynb` | created (new in branch) | Cells 0-3 rewritten |
| `scripts/build_notebook_cells_0_1.py` | created | Cell 0-1 builder |
| `scripts/build_notebook_cells_2_3.py` | created | Cell 2-3 builder |
| `.planning/REQUIREMENTS.md` | modified | QUERY-01 -> Phase 1 |

Note: "created (new in branch)" means the file pre-existed on disk in the main repo but was untracked by git; these are the first git-tracked copies of those files in this branch.

## Next Steps (Plan 01-02)

Plan 01-02 will wire OpusMT translation and query preprocessing:
- Use `OPUS_MT_DIR` path constant (already in Cell 0)
- Call `extract_legal_codes()` for D-03 query expansion (already defined in Cell 3)
- Call `canonicalize()` on translated queries (already defined in Cell 3)
- Call `tokenize_for_bm25_de()` to prepare query BM25 input (already defined in Cell 3)
- Replace the legacy Cells 4-7 (E5-small encode + naive retrieval + val eval) with the Phase 1 BGE-M3 + bm25s + RRF + mmarco pipeline across Plans 01-02 through 01-05.

## Self-Check: PASSED

Verified all created files exist:

- `notebook_kaggle.ipynb` — FOUND (8 cells, Cells 0-3 rewritten per plan)
- `requirements.txt` — FOUND (bm25s/CharSplit/nltk added)
- `download_models.py` — FOUND (BGE-M3 + opus-mt-tc-big-en-de added; Python AST valid)
- `scripts/build_notebook_cells_0_1.py` — FOUND
- `scripts/build_notebook_cells_2_3.py` — FOUND
- `.planning/REQUIREMENTS.md` — MODIFIED (QUERY-01 -> Phase 1 row present; no Phase 2 row for QUERY-01)

Verified all three task commits exist:

- `a6aaa38` — FOUND (Task 1: requirements + download_models + REQUIREMENTS)
- `8db77d8` — FOUND (Task 2: Cells 0-1)
- `570a0bf` — FOUND (Task 3: Cells 2-3)
