# LLM Agentic Legal Information Retrieval

## What This Is

A competitive Kaggle submission for the "LLM Agentic Legal Information Retrieval" competition. Given English legal questions about Swiss law, the system retrieves the most relevant Swiss legal citations (statutes and court decisions, mostly in German) and outputs them as submission.csv. The goal is to maximize Macro-F1 score and finish in the top 3 for prize money ($5K/$3K/$1K).

## Core Value

Retrieve correct Swiss legal citations across language boundaries — every missed citation is a direct hit to F1 score.

## Requirements

### Validated

- ✓ BM25 lexical retrieval on laws corpus — existing in `solution.py`
- ✓ Dense retrieval with multilingual-e5-large on laws corpus via FAISS — existing in `solution.py`
- ✓ RRF fusion of multiple retrieval signals — existing in `solution.py`
- ✓ Cross-encoder reranking with mMiniLM — existing in `solution.py`
- ✓ English-to-German query translation via OpusMT — existing in `solution.py`
- ✓ Legal code extraction from queries (Art. numbers, law codes) — existing in `solution.py`
- ✓ Per-query citation count calibration on validation set — existing in `solution.py`
- ✓ Kaggle/local runtime detection and conditional paths — existing in `solution.py`

### Active

- [ ] Fix GPU usage in Kaggle notebook (currently forced to CPU)
- [ ] Add BM25 retrieval over 2.47M court decisions corpus
- [ ] Dense reranking on top BM25 court candidates (sampled, not full corpus)
- [ ] Calibrate prediction count to ~20-30 per query instead of fixed 100
- [ ] Upgrade to multilingual-e5-large in Kaggle notebook (currently using e5-small)
- [ ] Entity-driven retrieval: parse explicit citations from queries, do direct lookup + graph expansion
- [ ] Citation graph expansion: if Art. X found, find BGE decisions mentioning Art. X
- [ ] Nearby article expansion (same law, nearby paragraph numbers)
- [ ] Mistral-7B 4-bit quantized for query analysis, citation generation, and query expansion in German legal language
- [ ] LLM-estimated citation count per query
- [ ] Fit entire pipeline within Kaggle 12-hour runtime on T4 x2 (no pre-uploaded datasets)
- [ ] Incremental development: each improvement layer submittable independently

### Out of Scope

- Pre-uploaded dense index as Kaggle dataset — user chose Kaggle-only, no external datasets
- Fine-tuning embedding models — not feasible within Kaggle runtime constraints
- Multi-language court corpus (FR/IT) — focus on German which covers the majority
- Creative prize track — focus is on leaderboard placement for cash prizes

## Context

**Competition:** Kaggle "LLM Agentic Legal Information Retrieval" — Swiss law citation retrieval. ~2 months remaining until close. 50% public / 50% private leaderboard split.

**Current score:** Val Macro-F1 = 0.009 (effectively zero). The submitted Kaggle notebook is a drastically simplified version of the local solution — dense-only, e5-small, laws-only, CPU-only.

**Critical data mismatches discovered:**
- Val queries expect 6x more citations than train queries (25.1 vs 4.1 mean)
- 41% of val citations are court decisions (BGE/docket), but current system predicts 0% court decisions
- Only 17.6% of val gold citations appear in training data — must do genuine corpus retrieval
- Cross-lingual challenge: English queries → German corpus

**Existing local solution (solution.py):** Has the right architecture (BM25 + dense + RRF + cross-encoder + translation) but was never successfully deployed on Kaggle. Will be refactored into the Kaggle notebook incrementally.

**Dataset sizes:**
- train.csv: 1,139 queries (German)
- val.csv: 10 queries (English)
- test.csv: 40 queries (English)
- laws_de.csv: 175,933 rows (German)
- court_considerations.csv: 2,476,316 rows (DE/FR/IT)

## Constraints

- **Runtime:** 12-hour limit on Kaggle offline notebook — no internet access
- **Hardware:** T4 x2 (16GB VRAM each) — must fit Mistral-7B 4-bit + e5-large + cross-encoder
- **VRAM:** Careful model lifecycle management needed — load/unload models sequentially, not all at once
- **Corpus size:** 2.47M court decisions — dense encoding infeasible for full corpus, BM25 + sampled dense is the strategy
- **Code competition:** Must produce submission.csv reproducibly from the notebook
- **No external data:** Everything runs within Kaggle — no pre-uploaded datasets

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Incremental build (quick wins → BM25 → entity → LLM) | Each layer is independently testable and submittable; tracks progress on leaderboard | — Pending |
| BM25 + sampled dense for court corpus | Full dense encoding of 2.47M docs exceeds time budget; BM25 top candidates then dense rerank is feasible | — Pending |
| Mistral-7B 4-bit for LLM augmentation | Best reasoning capability that fits T4 VRAM with quantization | — Pending |
| Refactor existing solution.py into Kaggle notebook | Local solution has correct architecture; avoid rewriting working components | — Pending |
| Kaggle-only (no pre-uploaded datasets) | Simplifies submission, avoids dataset management overhead | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-10 after initialization*
