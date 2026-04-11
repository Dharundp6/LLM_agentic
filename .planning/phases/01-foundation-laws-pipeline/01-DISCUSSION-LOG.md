# Phase 1: Foundation + Laws Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-11
**Phase:** 01-foundation-laws-pipeline
**Areas discussed:** Laws BM25 query language

---

## Gray Area Selection

Claude presented 4 candidate gray areas:

| Area | Selected |
|------|----------|
| Notebook refactor strategy | |
| Laws BM25 query language | ✓ |
| Reranker in Phase 1 | |
| Calibration approach | |

User selected **Laws BM25 query language** only. The other three areas were left
to Claude's discretion (see CONTEXT.md "Claude's Discretion" section).

---

## Laws BM25 Query Language

**Framing:** Laws corpus is German (`laws_de.csv`); val/test queries are English.
BGE-M3 dense handles cross-lingual natively, but BM25 is pure lexical — English
tokens will not match German corpus text. QUERY-01 (OpusMT EN→DE translation) is
officially mapped to Phase 2 in REQUIREMENTS.md, so Phase 1 needs a scoping call.

| Option | Description | Selected |
|--------|-------------|----------|
| Pull OpusMT forward to Phase 1 | Load `Helsinki-NLP/opus-mt-tc-big-en-de` in Phase 1, translate val/test EN→DE, feed German tokens to bm25s. Cleanest lexical match. Cost: second model upload, extra load/unload cycle, borrows QUERY-01 from Phase 2 scope. | ✓ |
| English + legal-code extraction | Keep Phase 1 translation-free. Tokenize English query + regex-extract codes (`Art. 221 StPO`, `OR`, `ZGB`) and append. Matches current solution.py pattern but with bm25s. | |
| Skip laws BM25 — dense only | Drop sparse signal in Phase 1; rely on BGE-M3 dense alone. Simplest but violates LAWS-03 / LAWS-04 and loses exact-code precision. | |
| BGE-M3 built-in sparse head | Use BGE-M3's own lexical output instead of bm25s. Single model, cross-lingual. Contradicts LAWS-03 German decompounding; no published benchmark on Swiss legal corpora. | |

**User's choice:** "delte the best i just want to win the competition" →
clarified as "select the best, I just want to win the competition."
User delegated the final pick to Claude with an explicit F1-first priority signal.

**Claude's selection:** Pull OpusMT forward to Phase 1 (Option 1).

**Rationale:**
1. Without translation, BM25 laws is effectively noise for the majority of val
   queries (those without explicit `Art. X` / law-code references), killing the
   sparse signal on which LAWS-03 / LAWS-04 depend.
2. OpusMT is small (~300 MB) and runs in seconds on ~50 queries total; VRAM cost
   is negligible when load/unloaded around BGE-M3 (FOUND-03 single-GPU invariant
   is preserved).
3. Legal code extraction (D-03) can still be layered on top of the German
   translation for exact-match precision — best of both.
4. Research PITFALLS CP-1 (German compound blindness) and LAWS-03 (German
   decompounding) both assume BM25 sees German input; the option is the only
   one that honors that assumption.
5. The cost — borrowing QUERY-01 one phase forward — is small and was confirmed
   by the user's F1-first priority signal.

**Follow-up decisions Claude locked without re-asking** (see CONTEXT.md D-02..D-05):
- Translate once at startup, cache per query, unload OpusMT before BGE-M3 loads
- Append `extract_legal_codes()` output to the German translation for BM25 input
- Feed BOTH English and translated German into BGE-M3 dense, average the embeddings
- Run canonicalization (QUERY-04) on both English and German forms before use

---

## Claude's Discretion

Areas where the user delegated — defaults recorded in CONTEXT.md `<decisions>`:

- **Notebook refactor strategy** — refactor `notebook_kaggle.ipynb` in place;
  keep `solution.py` as local dev mirror.
- **Reranker choice** — keep existing `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
  for Phase 1; defer jina-reranker-v2 swap to Phase 4 per FUSE-03.
- **Calibration approach** — reuse `calibrate_top_k()` grid search on val (K ∈ 1..80);
  per-query thresholding deferred (small val set → CP-4 overfitting risk).
- **FAISS install path** — `faiss-gpu-cu12` first, `faiss-cpu` fallback.
- **Default hyperparameters** — seed from solution.py (`BM25_LAWS_K=100`,
  `DENSE_LAWS_K=100`, `RRF_K_CONST=60`, `RERANK_K=150`); tune on val only.

## Deferred Ideas

- BGE-M3 built-in sparse head as primary lexical signal (rejected — contradicts
  LAWS-03 decompounding; no Swiss-legal benchmark)
- Per-query cross-encoder score-threshold calibration (deferred — CP-4 overfitting
  risk on 10-query val; LLM-based per-query K already scoped for Phase 5 LLM-04)
- Skipping laws BM25 entirely (rejected — loses exact-match precision signal)
- Jumping to jina-reranker-v2 in Phase 1 (deferred to Phase 4 per FUSE-03)
