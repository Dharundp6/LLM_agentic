# Phase 1: Foundation + Laws Pipeline - Research

**Researched:** 2026-04-11
**Domain:** GPU-enabled cross-lingual legal retrieval on Kaggle T4 — BGE-M3 dense + bm25s sparse over `laws_de.csv`, OpusMT query translation pulled forward from Phase 2, mmarco cross-encoder rerank, calibrated laws-only `submission.csv`
**Confidence:** MEDIUM-HIGH (stack and API paths verified; T4-specific VRAM/timing figures and bm25s German-decompounding recall are MEDIUM — must be smoke-tested in the notebook, not assumed from prior research)

---

## Summary

Phase 1 is a **stack swap on top of an existing monolithic notebook**, not a rewrite. The current `notebook_kaggle.ipynb` force-disables CUDA, uses `multilingual-e5-small`, indexes only the laws corpus, and falls back to a train-frequency baseline — it produces a structurally valid submission but is semantically broken. The shape of `solution.py` (a single file with load → index → retrieve → rerank → calibrate → write stages) is the right skeleton; Phase 1 rewires the model-backed stages (dense encoder, BM25 library, query translation) onto GPU with the correct versions, the correct pooling, the correct prefixes, and the correct hard-fail guards.

The non-obvious correctness cliffs the planner must explicitly prevent — and that no verification step currently catches — are:

1. **BGE-M3 uses CLS pooling, NOT mean pooling.** The `mean_pool()` helper at `solution.py:127-129` (still carried into the notebook) is wrong for BGE-M3 and will silently degrade dense recall. Use `last_hidden_state[:, 0]` + L2 normalization. See `[CITED: huggingface.co/BAAI/bge-m3/discussions/17]` and BGE official docs. This is not a "nice optimization"; CLS vs mean changes what the model was trained to output.
2. **BGE-M3 does NOT use `"query: "` / `"passage: "` prefixes.** Those are e5-family prefixes. BGE-M3 takes raw text for both sides. Porting the e5 `encode()` prefix=... parameter unchanged will poison every embedding. `[CITED: huggingface.co/BAAI/bge-m3/discussions/35]`
3. **`faiss-gpu-cu12` is Linux x86_64 only.** `[CITED: pypi.org/project/faiss-gpu-cu12/]` It works on Kaggle's T4 kernel, but the local dev environment in this project is Windows — the notebook must support `faiss-cpu` on local and `faiss-gpu-cu12` on Kaggle, detected at import time. `solution.py`'s `IS_KAGGLE` switch already has the right shape; extend it.
4. **bm25s German decompounding is not built-in.** bm25s supports NLTK German stopwords via `stopwords="de"` and accepts a `stemmer` callable, but compound splitting is a separate library (CharSplit or `german_compound_splitter`). CP-1/LAWS-03 says "decompound before BM25" — this requires an explicit pre-tokenization pass, not a bm25s flag.
5. **Laws corpus is 175K rows but each row has a German legal text — encoding time on T4 for 175K × max_length=512 with fp16 is 3-6 minutes in practice for BGE-M3 560M params.** The 60-second checkpoint from D-07 should require **≥1000 docs encoded at 60s**, NOT "fully done at 60s." At 64/batch × 1-2 batches/sec on T4, 1000 docs is a reasonable floor that catches CPU fallback without false-alarming on healthy GPU runs.

**Primary recommendation:** Rewrite `notebook_kaggle.ipynb` in place as a linear-cell Kaggle notebook that: (1) hard-asserts CUDA at cell 1, (2) loads OpusMT → translates all val+test queries → unloads, (3) loads BGE-M3 fp16 → encodes laws corpus with CLS pooling + raw text + L2 normalize → builds FAISS IndexFlatIP → encodes queries (EN and DE averaged) → kNN → unloads, (4) builds bm25s laws index with decompounded German tokens → scores translated queries with appended legal codes → kNN, (5) RRF-fuses dense + sparse → reranks top-150 with mmarco → calibrates K on val → validates and writes `submission.csv`. Keep `solution.py` as a local-dev mirror but don't require byte-parity.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Pull OpusMT (`Helsinki-NLP/opus-mt-tc-big-en-de`) forward from Phase 2 into Phase 1. BM25 laws queries are the German translation of the English input, not the raw English text. QUERY-01 must be re-mapped from Phase 2 to Phase 1 in the traceability table when the planner updates REQUIREMENTS.md.
- **D-02:** Translate once at startup, cache per query in a `dict[query_id → german_translation]`. Load OpusMT first, translate all val+test queries in a single batch, then `del` OpusMT and `cuda.empty_cache()` **before** BGE-M3 loads. Only ~50 queries total — translation completes in seconds.
- **D-03:** Append regex-extracted legal codes (`extract_legal_codes()` from `solution.py:319-330`) to the translated German query before BM25 tokenization. Extract codes from the **original English query** (codes are language-agnostic). Final BM25 input = `{german_translation} {extracted_codes}`.
- **D-04:** Dense path receives BOTH the raw English query and the German translation (QUERY-03). BGE-M3 embeds each, the two embeddings are averaged and L2-renormalized, then queried once against the FAISS index.
- **D-05:** Query canonicalization (QUERY-04) runs on BOTH the English original and the German translation before ALL downstream use (code extraction, BM25 tokenization, dense encoding). Normalize whitespace around `Art.` and `Abs.`, normalize Roman vs Arabic article numerals, normalize `ss` ↔ `ß`, normalize `ZGB` ↔ `CC` aliasing. Same canonicalizer is used for laws text preprocessing (LAWS-05) and final submission citation formatting (CALIB-02) so the pipeline is self-consistent end to end.
- **D-06:** Strict single-GPU-resident model invariant. Order: `(1) load OpusMT → translate → del + empty_cache`, `(2) load BGE-M3 fp16 → encode laws corpus → build FAISS → encode queries → kNN → del + empty_cache`, `(3) load cross-encoder → rerank → del + empty_cache`. No two neural models coexist on GPU at any time.
- **D-07:** CUDA assert is the first executable cell (FOUND-01 / AH-1): `assert torch.cuda.is_available()`, log `torch.cuda.get_device_name(0)`, log VRAM total. Add a 60-second encoding checkpoint: if fewer than 1000 laws docs encoded after 60 s, raise and abort.

### Claude's Discretion

- Notebook vs `solution.py` source of truth: default to refactoring `notebook_kaggle.ipynb` as canonical Kaggle artifact; keep `solution.py` as local dev mirror only. No byte-parity required.
- Cross-encoder choice for Phase 1: default to existing `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Jina v2 is locked for Phase 4 (FUSE-03) — do NOT swap here.
- Calibration approach: default to existing `calibrate_top_k()` grid search over K ∈ [1, 80] on val. Per-query thresholds are scoped for Phase 5 (LLM-04); CP-4 overfitting risk on 10-query val blocks aggressive tuning here.
- FAISS install path: try `faiss-gpu-cu12` first; fall back to `faiss-cpu` on install failure. 175K × 1024 IndexFlatIP is fine on CPU if needed.
- Hyperparameter starting points: `BM25_LAWS_K=100`, `DENSE_LAWS_K=100`, `RRF_K_CONST=60`, `RERANK_K=150`. Tune on val only.
- Legal code extraction regex coverage: extend the existing `QUERY_CODES` set as needed; adding law codes is safe.

### Deferred Ideas (OUT OF SCOPE)

- BGE-M3 built-in sparse head as primary lexical signal — rejected for Phase 1 (contradicts LAWS-03's explicit German decompounding requirement, no Swiss legal benchmark).
- Per-query cross-encoder score threshold for calibration — rejected for Phase 1 (val too small, overfitting risk; LLM-04 in Phase 5 is the principled route).
- Skipping BM25 laws entirely — rejected (throws away guaranteed-precision exact-match hits).
- Jumping directly to jina-reranker-v2 — deferred to Phase 4 per FUSE-03.
</user_constraints>

## Project Constraints (from CLAUDE.md)

Extracted from `./CLAUDE.md` and the parent `C:\Users\Dharun prasanth\CLAUDE.md`. The planner must respect these as hard rules, not suggestions:

- **Kaggle runtime is 12 hours, offline, no internet**, T4 x2 (16GB VRAM each), ~30GB RAM. Any plan step that requires network access during kernel execution is invalid.
- **Monolithic notebook style** — do NOT introduce `src/` packages or import layering. `solution.py` and `notebook_kaggle.ipynb` are both single-file by design; Phase 1 stays single-file.
- **Sonnet default model** for all routine code work; reserve Opus only for complex multi-file reasoning (the parent CLAUDE.md rule).
- **No "Claude"/"Anthropic"/AI attribution in commit messages, PR titles, or branch names.** `gsd-tools.cjs commit` should be invoked with a plain message. No `Co-Authored-By: Claude` footers.
- **GSD workflow enforcement** — file edits must go through a GSD command; Phase 1 is already inside `/gsd-execute-phase`, so this is satisfied when the executor runs.
- **Assertion-based validation, no try/except.** The existing codebase pattern is `assert X, "reason"`; do not introduce try/except scaffolding. FOUND-01 hard assert is an instance of this convention, not an exception to it.
- **Index-first MD reading** — when the planner needs to re-load docs, read `.planning/phases/01-foundation-laws-pipeline/01-CONTEXT.md` + this RESEARCH.md + `solution.py` first; avoid re-reading all nine `.planning/` docs.

## Phase Requirements

The planner MUST produce a traceable implementation for each of the 22 IDs below (21 native + QUERY-01 pulled forward). Every PLAN task should cite one or more REQ IDs in its rationale.

| ID | Description | Research Support |
|----|-------------|------------------|
| FOUND-01 | `torch.cuda.is_available()` asserted as first action | See §BGE-M3 on Kaggle T4, §Common Pitfalls AH-1. Cell-1 assert block with device name + VRAM log + 60s encoding checkpoint. |
| FOUND-02 | All neural models confirmed on CUDA with logged device name | See §Model Lifecycle. Every load helper prints `next(model.parameters()).device` after `.to(DEVICE)`. |
| FOUND-03 | Sequential model lifecycle, one model on GPU at a time, explicit `empty_cache()` between stages | See §Model Lifecycle (D-06). Explicit `del tok; del mdl; gc.collect(); torch.cuda.empty_cache()` after each of OpusMT / BGE-M3 / cross-encoder stages. |
| FOUND-04 | Kaggle/local runtime detection with conditional paths | Extend `IS_KAGGLE` block at `solution.py:20-35` to include `BGE_M3_DIR`, `OPUS_MT_DIR`. See §Kaggle Path Layout. |
| FOUND-05 | End-to-end smoke test on 3 val queries in <5 minutes | See §Smoke Test Architecture. `SMOKE=True` flag short-circuits laws encoding to first 5000 docs; runs full retrieve→rerank→F1 on val queries [0:3]. |
| FOUND-06 | Full pipeline completes in <11 hours (1h safety margin) | See §Runtime Budget. Expected ~10 min laws encoding + ~1 min bm25s build + ~1 min per-query retrieve+rerank × 50 queries ≈ 1 hour total. Margin is large for Phase 1; shrinks in later phases. |
| LAWS-01 | BGE-M3 replaces multilingual-e5-large, fp16 | See §BGE-M3 Loading. Raw `transformers.AutoModel` path preferred over FlagEmbedding (fewer moving parts; transformers is already in requirements.txt). CLS pool + L2 normalize. |
| LAWS-02 | FAISS `IndexFlatIP` over all 175K laws with L2-normalized embeddings | See §FAISS Index Strategy. 175K × 1024 × 4 bytes = ~730 MB. GpuIndexFlat fits trivially; CPU fallback also fine. |
| LAWS-03 | bm25s sparse index with German decompounding, no stemming | See §bm25s Integration. Pre-tokenize with `tokenize_for_bm25()` + CharSplit decompounding + NLTK German stopword removal → pass list-of-list tokens to `bm25s.BM25().index()`. NO Snowball stemmer (GerDaLIR guidance: stemming hurts German legal BM25). |
| LAWS-04 | Laws sub-pipeline returns dense top-K + BM25 top-K per query | See §retrieve_candidates rewrite. Two ranked lists: `dense_l_ids` and `bm25_l_ids`. No court list in Phase 1 (that's Phase 2). |
| LAWS-05 | Laws corpus text preprocessed with canonical Swiss citation format | See §Swiss Citation Canonicalization. Single `canonicalize()` function applied to `laws['citation']` column before indexing AND to predicted citations before writing `submission.csv`. |
| QUERY-01 | OpusMT EN→DE translation (pulled forward from Phase 2 per D-01) | See §OpusMT Loading. `Helsinki-NLP/opus-mt-tc-big-en-de`, batched generate, unload before BGE-M3 loads. |
| QUERY-03 | BGE-M3 receives both EN and DE query forms | See §Dual-Query Dense Retrieval. Encode `[q_en_canonical, q_de_canonical]` as two texts, average the two 1024-dim vectors, re-normalize, search FAISS once. |
| QUERY-04 | Query canonicalization before entity matching | See §Swiss Citation Canonicalization. Same `canonicalize()` used for queries and corpus text. |
| CALIB-01 | Per-query citation count calibrated on val, target ~20-30 | See §Calibration. Reuse `calibrate_top_k()`; expand range to [1, 60] to give room for val's 25-mean distribution. |
| CALIB-02 | Final predictions canonicalized to exact Swiss format | See §Swiss Citation Canonicalization. Applied inside `submission.csv` writer. |
| CALIB-03 | `submission.csv` format validated before writing | See §Submission Validator. Assertion block: row count == test row count, all `query_id`s present, no NaN in `predicted_citations`, semicolons as separator, no trailing semicolons, no commas inside citation strings. |
| CALIB-04 | Val Macro-F1 computed locally after each change | See §Calibration. Reuse `macro_f1()`; log `*** Val macro-F1 = X.XXXX @ top-K ***` banner matching existing pattern. |
| CALIB-05 | Calibration never touches test | See §Calibration. Enforced by `calibrate_top_k(val_results)` signature — test set is not passed in. |
| INC-01 | Phase 1 produces an independently submittable submission | Satisfied by this phase's output contract; see §Incremental Submission Contract. |
| INC-02 | Feature flags allow disabling later-phase components | Add `USE_RERANKER=True`, `USE_LLM_AUGMENTATION=False`, `USE_COURT_CORPUS=False` module-level flags. Later phases flip them without touching logic. |
| INC-03 | Git tags mark each phase's submission-ready state | Plan must include a final task: `git tag phase-1-submission` after val F1 is logged and submission.csv passes validation. Tag is local; push is optional per CLAUDE.md (user-gated). |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | 2.1.0+ | Neural model inference on CUDA | Already in requirements.txt; `.planning/research/STACK.md` pins 2.1+. Kaggle T4 kernels ship with torch ~2.4-2.6 depending on kernel date `[VERIFIED: solution.py imports + PyPI]` |
| `transformers` | 4.38.0+ | BGE-M3, OpusMT, mmarco loading via `AutoModel` / `MarianMTModel` / `AutoModelForSequenceClassification` | Already in requirements.txt; BGE-M3 requires 4.33+ for the native config; 4.38 is safe floor `[VERIFIED: existing requirements.txt]` |
| `BAAI/bge-m3` (HF model) | latest (model is stable since 2024-Q1) | Dense embeddings, 1024-dim, 8192-token context, cross-lingual EN↔DE `[CITED: huggingface.co/BAAI/bge-m3]` | Outperforms e5-large on MIRACL; dense+sparse+ColBERT single checkpoint; Phase 1 uses dense mode only `[CITED: .planning/research/STACK.md]` |
| `bm25s` | 0.2.x (0.2.0 confirmed on PyPI; post-0.2 adds Numba backend for ~2x speedup) `[CITED: pypi.org/project/bm25s/]` | Fast BM25 over laws (175K docs) and later court (2.47M) | 500x faster than rank-bm25; memory-mapped index; same scoring semantics as rank-bm25 `[CITED: arxiv.org/abs/2407.03618]` |
| `faiss-gpu-cu12` | 1.14.1.post1 (released 2026-03-07) `[VERIFIED: pypi.org/project/faiss-gpu-cu12/]` | GPU IndexFlatIP over laws | 10-50× faster than CPU FAISS for exact search at 175K × 1024 `[CITED: .planning/research/STACK.md]` |
| `faiss-cpu` | 1.7.4+ | Fallback when `faiss-gpu-cu12` install fails or on Windows local dev | Already in requirements.txt; 175K IndexFlatIP is fine on CPU (<1s per query) `[VERIFIED: requirements.txt]` |
| `Helsinki-NLP/opus-mt-tc-big-en-de` (HF model) | latest | EN→DE translation of ~50 queries | Upgrade over `opus-mt-en-de` (still ~298MB-class); better quality on domain content; same MarianMT API `[CITED: .planning/research/STACK.md]` |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (HF model) | latest | Rerank top-150 candidates | Already in `models/`; 120MB; Phase 1 uses as-is, Phase 4 swaps for jina v2 `[VERIFIED: existing models/ directory]` |
| `sentencepiece` | 0.1.99+ | Required by MarianMT and BGE-M3 tokenizers | Already in requirements.txt `[VERIFIED: requirements.txt]` |
| `sacremoses` | 0.0.53+ | Required by MarianMT detokenization | Already in requirements.txt `[VERIFIED: requirements.txt]` |
| `numpy`, `pandas` | 1.24+, 2.0+ | Array and DataFrame I/O | Already present `[VERIFIED: requirements.txt]` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `CharSplit` (pypi: `compound-split` or `CharSplit`) | 0.x | German compound word decompounding for bm25s input `[CITED: github.com/dtuggener/CharSplit]` | Wrap inside `tokenize_for_bm25_de()`; append split heads as extra tokens (don't replace original compound, augment) |
| `nltk` | 3.8+ | German stopword list (`stopwords.words("german")`) for bm25s filtering `[CITED: nltk.org]` | Import once at top of notebook; cache list to a frozenset |
| `gc` (stdlib) | — | Explicit `gc.collect()` after `del model` to force heap release | After every model unload |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `transformers.AutoModel` load | `FlagEmbedding.BGEM3FlagModel` | FlagEmbedding wraps the load + pooling + multi-head output in one call `[CITED: huggingface.co/BAAI/bge-m3]`. Pros: one-line load, handles dense+sparse+ColBERT together. Cons: adds a dependency that isn't in requirements.txt, wraps the HF model and makes debugging pooling issues harder, and Phase 1 only needs dense mode. **Recommendation: raw transformers.** Phase 4/5 may revisit if the sparse head becomes a candidate signal. |
| CLS pooling via `outputs.pooler_output` | CLS pooling via `outputs.last_hidden_state[:, 0]` | `pooler_output` applies an additional tanh projection layer that BGE-M3 was NOT trained to use as the sentence embedding `[CITED: huggingface.co/BAAI/bge-m3/discussions/17 and /80]`. **Use `last_hidden_state[:, 0]` + L2 normalize.** This is a correctness issue, not a preference. |
| bm25s stopwords via `stopwords="de"` string arg | Pre-filtered list-of-list tokens | bm25s's `stopwords="de"` works off NLTK but the exact list ISO code is "german" not "de" in NLTK; passing the wrong code silently gives an empty stopword list. **Safer: pre-filter in our own tokenizer**, pass clean list-of-lists to `bm25s.BM25().index()`. |
| `opus-mt-tc-big-en-de` on GPU | opus-mt-tc-big on CPU | The model is ~1.2GB and 50 short queries translate in seconds on CPU `[CITED: .planning/research/STACK.md]`. But D-02 specifies GPU load with explicit unload — honor the locked decision. GPU load is simpler code (one DEVICE path) and the cost is ~3 seconds. |
| `faiss-gpu-cu12` on Kaggle | `faiss-cpu` on Kaggle | GPU IndexFlatIP is 10-50× faster than CPU, but 175K × 1024 laws × 50 queries on CPU is still <5 seconds total. **If `faiss-gpu-cu12` install fails in the offline kernel, `faiss-cpu` is a zero-risk fallback** with no meaningful latency cost. |

**Installation (Kaggle notebook cell 0, before any other imports):**

```python
# Kaggle kernel pre-install cell. Pinned, fail-loud. Runs only on Kaggle (IS_KAGGLE check).
import subprocess, sys, os
IS_KAGGLE = os.path.exists("/kaggle")
if IS_KAGGLE:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "bm25s>=0.2.0"])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "faiss-gpu-cu12>=1.14"])
        USE_FAISS_GPU = True
    except subprocess.CalledProcessError:
        print("faiss-gpu-cu12 install failed; falling back to faiss-cpu", flush=True)
        USE_FAISS_GPU = False
    # CharSplit + NLTK german stopwords
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "CharSplit", "nltk"])
    import nltk; nltk.download("stopwords", quiet=True)
else:
    USE_FAISS_GPU = False  # local Windows dev always CPU faiss
```

**Version verification (run once before committing the cell):**

```bash
# On a machine with internet, verify the pins. Record output in the commit message.
python -m pip index versions bm25s
python -m pip index versions faiss-gpu-cu12
python -c "import transformers; print(transformers.__version__)"
```

At research time (2026-04-11):
- `faiss-gpu-cu12` — 1.14.1.post1 (2026-03-07) `[VERIFIED: pypi.org/project/faiss-gpu-cu12/]`
- `bm25s` — 0.2.x (Numba backend post-0.2.0) `[CITED: github.com/xhluca/bm25s]`
- `transformers` — 4.38+ floor; Kaggle kernels typically ship 4.44-4.50 `[ASSUMED: not verified in kernel at research time]`

---

## Architecture Patterns

### Recommended Notebook Cell Structure

The notebook is a single linear script expressed as Jupyter cells. Cell boundaries exist for runtime-checkpointing (re-running one cell without re-running the whole thing) and for the Kaggle reviewer to see progress markers. Do NOT add cross-cell class hierarchies or abstraction layers.

```
Cell 0  — pip installs (Kaggle only), IS_KAGGLE flag, path constants
Cell 1  — HARD ASSERT: torch.cuda.is_available(), device name, VRAM; seeds; imports
Cell 2  — Load CSVs (laws, val, test, train); log shapes and memory
Cell 3  — Canonicalization helpers (canonicalize, extract_legal_codes, tokenize_for_bm25_de)
Cell 4  — Stage 1: OpusMT load → translate all val+test queries → cache dict → unload + empty_cache
Cell 5  — Stage 2a: BGE-M3 load (fp16) + 1024-dim assert + CLS-pool + L2 normalize
Cell 6  — Stage 2b: Encode laws corpus (with 60s/1000-doc checkpoint) → build FAISS index
Cell 7  — Stage 2c: Encode val+test queries (EN and DE averaged) → kNN → store per-query top-K dense IDs → unload BGE-M3
Cell 8  — Stage 3: bm25s laws index build → score translated+code-augmented queries → top-K sparse IDs
Cell 9  — Stage 4: RRF fuse dense+sparse → top-RERANK_K candidates per query
Cell 10 — Stage 5: Load cross-encoder → batch rerank → unload
Cell 11 — Stage 6a: Compute val Macro-F1 → calibrate K → log best_k
Cell 12 — Stage 6b: Apply best_k to test → canonicalize → validate submission format → write submission.csv
Cell 13 — Stage 6c: SMOKE block gated by SMOKE=True; runs cells 5-12 on laws[:5000] and val[:3]
```

**Locked:** cells run strictly top-to-bottom; no cell mutates state in a prior cell. Each cell's outputs are the inputs to the next. This is enforced by convention (no class-level state) and by VRAM unload points.

### Pattern 1: Hard-Fail CUDA Assert Cell

**What:** The very first executable cell blocks the notebook if CUDA is unavailable or the 60-second encoding checkpoint fails. No silent CPU fallback.
**When to use:** Cell 1, every run. Even for SMOKE mode.

```python
# Cell 1 — FAIL LOUDLY IF GPU IS NOT READY
import torch, time, gc
import numpy as np, random

# Hard gate: abort if GPU is not present. No silent CPU fallback.
assert torch.cuda.is_available(), (
    "GPU unavailable. Check Kaggle Settings → Accelerator → GPU T4 x2. "
    "AH-1 prevention: do not proceed on CPU."
)

DEVICE = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"VRAM free:  {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

# Deterministic seeds (MP-2)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

### Pattern 2: BGE-M3 Loading with Correct Pooling

**What:** Load BGE-M3 via raw `transformers.AutoModel`, fp16 cast, use `last_hidden_state[:, 0]` for CLS pooling, L2 normalize, assert embedding dim == 1024.
**When to use:** Cell 5, exactly once per run.

```python
# Cell 5 — BGE-M3 load with correct pooling. CLS pool, not mean pool.
from transformers import AutoTokenizer, AutoModel

BGE_M3_DIR = Path("/kaggle/input/bge-m3") if IS_KAGGLE else "BAAI/bge-m3"
print(f"Loading BGE-M3 from {BGE_M3_DIR}")
t0 = time.time()
bge_tok = AutoTokenizer.from_pretrained(str(BGE_M3_DIR))
bge_mdl = AutoModel.from_pretrained(str(BGE_M3_DIR), torch_dtype=torch.float16)
bge_mdl = bge_mdl.to(DEVICE).eval()
print(f"  loaded in {time.time()-t0:.1f}s; "
      f"params={sum(p.numel() for p in bge_mdl.parameters())/1e6:.0f}M; "
      f"dtype={next(bge_mdl.parameters()).dtype}; "
      f"device={next(bge_mdl.parameters()).device}")

@torch.no_grad()
def bge_encode(texts, batch_size=64, max_length=512):
    """
    BGE-M3 dense encoder. CLS pooling + L2 normalize. No prefixes.
    Returns (N, 1024) float32 numpy array.
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t)[:2000] for t in texts[i:i+batch_size]]  # soft char cap
        enc = bge_tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(DEVICE)
        out = bge_mdl(**enc)
        # CLS pool — the first token of last_hidden_state. NOT mean pool.
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.float().cpu().numpy())
    return np.vstack(all_embs).astype("float32")

# Smoke-test: encode two short strings and assert dim == 1024
_probe = bge_encode(["Art. 1 ZGB", "Article 1 Swiss Civil Code"])
assert _probe.shape == (2, 1024), f"BGE-M3 dim mismatch: {_probe.shape}"
print(f"  probe dim OK: {_probe.shape}")
```

**Sources:**
- CLS pooling: `[CITED: huggingface.co/BAAI/bge-m3/discussions/17]`, `[CITED: huggingface.co/BAAI/bge-m3/discussions/80]`
- No prefixes: `[CITED: huggingface.co/BAAI/bge-m3/discussions/35]`
- Dimension 1024: `[CITED: huggingface.co/BAAI/bge-m3]`

### Pattern 3: 60-Second Encoding Checkpoint

**What:** During the laws corpus encode loop, snapshot wall-clock time at start. After 60 s, assert that at least 1000 docs have been encoded. Abort otherwise — this is the canary for silent CPU fallback where `torch.cuda.is_available()` returns True but kernels fall back to CPU due to library mismatches (CUDA driver/torch mismatch, bitsandbytes kernel compile failure, etc.).
**When to use:** Inside the laws encoding loop (cell 6).

```python
# Cell 6 — Encode laws corpus with the AH-1 checkpoint
laws_texts = [
    canonicalize(f"{row['citation']} {str(row['text'])[:1500]}")
    for _, row in laws.iterrows()
]
print(f"Encoding {len(laws_texts):,} laws docs...")

t_start = time.time()
checkpoint_fired = False
all_embs = []
BATCH = 64
for i in range(0, len(laws_texts), BATCH):
    batch = laws_texts[i:i+BATCH]
    enc = bge_tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = bge_mdl(**enc)
    emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
    all_embs.append(emb.float().cpu().numpy())

    # AH-1 checkpoint: after 60s of wall-clock, require >=1000 encoded
    if not checkpoint_fired and time.time() - t_start > 60:
        assert (i + BATCH) >= 1000, (
            f"AH-1 TRIGGER: only {i+BATCH} docs encoded in 60s. "
            f"Silent CPU fallback suspected. Aborting run."
        )
        checkpoint_fired = True
        print(f"  60s checkpoint OK: {i+BATCH} docs encoded, proceeding")

    if (i // BATCH) % 200 == 0 and i > 0:
        elapsed = time.time() - t_start
        rate = (i + BATCH) / elapsed
        eta = (len(laws_texts) - i - BATCH) / rate
        print(f"  {i+BATCH:,}/{len(laws_texts):,}  [{elapsed:.0f}s, "
              f"{rate:.0f} doc/s, ETA {eta:.0f}s]", flush=True)

laws_embs = np.vstack(all_embs).astype("float32")
print(f"Encoded laws: {laws_embs.shape}  [{time.time()-t_start:.1f}s total]")
del all_embs; gc.collect()
```

### Pattern 4: Dual-Query Dense Retrieval (EN + DE averaged)

**What:** Per D-04, the dense signal averages BGE-M3 embeddings of the raw English query and the German translation (canonicalized). L2-normalize after averaging.
**When to use:** Cell 7, per query.

```python
# Per query: encode both forms and average
def dense_query_embedding(q_en_canon, q_de_canon):
    pair = bge_encode([q_en_canon, q_de_canon])  # shape (2, 1024)
    avg = pair.mean(axis=0, keepdims=True)       # shape (1, 1024)
    norm = np.linalg.norm(avg, axis=1, keepdims=True)
    return (avg / np.maximum(norm, 1e-9)).astype("float32")
```

Note: averaging two already-normalized unit vectors and re-normalizing is geometrically equivalent to taking their midpoint on the unit sphere. This is a common and well-behaved operation; it does NOT require any additional weighting for Phase 1.

### Pattern 5: bm25s with Pre-tokenized Input

**What:** Pass pre-decompounded German tokens directly to bm25s. Do NOT use `bm25s.tokenize()` as the primary path — it assumes generic whitespace + optional Snowball stemmer. We want custom legal-aware tokenization + CharSplit decompounding + NLTK German stopword removal.
**When to use:** Cell 8.

```python
# Cell 8 — bm25s laws index with custom German tokenization
import bm25s
from nltk.corpus import stopwords as nltk_stopwords
from charsplit import Splitter  # pip install charsplit OR compound-split

GERMAN_STOP = frozenset(nltk_stopwords.words("german"))
_splitter = Splitter()

def tokenize_for_bm25_de(text: str) -> list[str]:
    """
    Legal-aware German tokenization for bm25s.
    1. Lowercase, split on word+dot boundaries (preserves Art., Abs., BGE, etc.)
    2. Remove German stopwords (der, die, das, gemäß, ...)
    3. Decompound long tokens with CharSplit; include BOTH the original token
       AND the head-split components (augmentation, not replacement).
    4. Normalize ss/ß already handled upstream by canonicalize().
    """
    text = str(text).lower()
    raw = re.findall(r"[\w.]+", text)
    out = []
    for tok in raw:
        if tok in GERMAN_STOP:
            continue
        if len(tok) <= 3 or "." in tok:
            out.append(tok)
            continue
        out.append(tok)  # keep original
        # CharSplit returns list of (score, head, tail) tuples
        splits = _splitter.split_compound(tok)
        if splits:
            score, head, tail = splits[0]
            if score > 0.5 and head and tail:
                out.append(head.lower())
                out.append(tail.lower())
    return out

# Build index: pre-tokenize list-of-list, then pass to bm25s
print("Building bm25s laws index...")
t0 = time.time()
laws_bm25_texts = [
    canonicalize(f"{row['citation']} {row['text']}")
    for _, row in laws.iterrows()
]
laws_bm25_tokens = [tokenize_for_bm25_de(t) for t in laws_bm25_texts]
print(f"  tokenized {len(laws_bm25_tokens):,} laws  [{time.time()-t0:.1f}s]")

t0 = time.time()
bm25_laws = bm25s.BM25()
bm25_laws.index(laws_bm25_tokens)  # list-of-list input is accepted
print(f"  indexed                        [{time.time()-t0:.1f}s]")
```

**Sources:**
- bm25s accepts list-of-list-of-tokens in `index()`: `[CITED: github.com/xhluca/bm25s]`
- CharSplit API: `splitter.split_compound(word)` returns ranked splits `[CITED: github.com/dtuggener/CharSplit]`
- German stopword list via NLTK: `nltk.corpus.stopwords.words("german")` `[CITED: nltk.org]`
- GerDaLIR guidance against stemming on German legal text: `[CITED: .planning/research/PITFALLS.md §CP-1]`

**Known uncertainty:** CharSplit's exact PyPI name varies (`CharSplit`, `charsplit`, `compound-split`, `charsplit-nonlinear`). **[ASSUMPTION A5]** The planner must pick one on install and verify the API matches the example. `german_compound_splitter` (via pyahocorasick) is the alternative if CharSplit install fails on the Kaggle kernel.

### Pattern 6: Sequential Model Lifecycle (D-06)

**What:** Explicit unload after each stage. Never keep two models on GPU.
**When to use:** Between every cell that loads a model.

```python
def unload(*objs):
    """Free GPU memory: del references, gc collect, cuda empty cache."""
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"  unloaded → VRAM free: {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

# End of OpusMT stage:
translations = {qid: de for qid, de in zip(all_query_ids, opus_outputs)}
unload(opus_tok, opus_mdl)

# End of BGE-M3 stage:
# (FAISS index lives on CPU in RAM, not GPU, so it survives the unload)
unload(bge_tok, bge_mdl)

# End of cross-encoder stage:
unload(rerank_tok, rerank_mdl)
```

### Anti-Patterns to Avoid

- **Mean-pooling BGE-M3** — wrong pooling for this model family; destroys recall silently. Use CLS pool.
- **Adding `"query: "` / `"passage: "` prefixes to BGE-M3** — those are e5 prefixes; BGE-M3 was trained without them. Feeding prefixes poisons every embedding.
- **Keeping all three models (OpusMT + BGE-M3 + cross-encoder) on GPU simultaneously** — violates D-06; risks OOM under T4's 16GB budget with activation memory + FAISS GPU index headroom.
- **Silent try/except around the CUDA assert** — defeats AH-1 prevention. Fail loudly.
- **Calibrating K on train instead of val** — AH-5 (train mean 4.1 cites, val mean 25.1 cites — 6× distribution mismatch). CALIB-05 is explicit: val only.
- **Returning a fixed 100 predictions per query** — AH-3. Always gate on `best_k` from calibration.
- **Writing `submission.csv` without format validation** — MP-5 silent rejection. Always validate before writing.
- **Using `calibrate_top_k()` range [1, 80] with val mean at 25** — fine in principle but the search boundary hits would overfit. Widen to [1, 60] and log a warning if `best_k` lands on the boundary (CP-4 warning sign).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BGE-M3 CLS pooling | Custom CLS extraction from `last_hidden_state` | Standard `outputs.last_hidden_state[:, 0]` + `F.normalize` | The one-liner IS the pattern; don't wrap it in a class or add tanh/dropout layers "for safety." BGE's training pipeline is deterministic here. |
| German compound splitting | Hand-rolled substring splitter | CharSplit (probability model trained on 1M German Wikipedia nouns, ~95% head-detection accuracy `[CITED: github.com/dtuggener/CharSplit]`) or `german_compound_splitter` (Aho-Corasick + dictionary) | Hand-rolled rules fail on Swiss German spellings, Fugen-s/Fugen-n infixes, and hyphenated compounds. Libraries already handle these. |
| Reciprocal rank fusion | Custom weighted sum | `rrf_fuse()` at `solution.py:220-226` (already in codebase) | The existing helper is 6 lines and matches published RRF semantics (1/(k+rank+1) with k=60). Phase 1 has only 2 lists (dense + sparse), which is the simplest case. |
| BM25 indexing | Pure Python loops à la rank-bm25 | `bm25s.BM25().index(tokens)` | rank-bm25 is 500× slower; even for 175K laws, `bm25s` builds in <2 seconds and cleanly extends to 2.47M court in Phase 2. `[CITED: arxiv.org/abs/2407.03618]` |
| FAISS IndexFlat CPU→GPU transfer | Manual wrapping with `StandardGpuResources` | `faiss.index_cpu_to_gpu(res, 0, index)` OR just build directly on CPU | 175K × 1024 IndexFlatIP takes <5s on CPU for all val+test queries. GPU is a speedup, not a requirement. **Recommend CPU IndexFlatIP in Phase 1 unless wall-clock encoding + retrieval exceeds the smoke-test budget.** |
| MarianMT tokenization | Manual sentencepiece handling | `AutoTokenizer.from_pretrained(opus_dir)` — returns a MarianTokenizer instance | `AutoTokenizer` dispatches correctly for Helsinki-NLP/opus-mt-tc-big-* `[CITED: huggingface.co/docs/transformers/model_doc/marian]`. No need to import `MarianTokenizer` explicitly. |
| Submission CSV format validation | Hand-rolled schema check | A dedicated `validate_submission(df, test_df)` function with explicit asserts | One-function, one-purpose, one-place-to-audit. See Pattern below. |
| Swiss citation canonicalization | Ad-hoc `str.replace` chains scattered across the codebase | A single `canonicalize(s)` function used by corpus-load, query-process, and submission-write paths | Consistency bug here → silent false negatives. See §Swiss Citation Canonicalization. |

**Key insight:** Phase 1 is about *composing existing library calls correctly* under tight VRAM and runtime constraints. Every "custom utility" is a candidate for a silent bug that later phases will inherit. Push all non-trivial logic into named, single-purpose helper functions so they can be unit-checked in the smoke cell.

---

## Swiss Citation Canonicalization (CP-6 / QUERY-04 / CALIB-02 / LAWS-05)

The canonical format is dictated by the **actual strings in `laws_de.csv` and `court_considerations.csv`**. Kaggle uses exact string matching `[CITED: AboutData.md]`, so the corpus strings ARE the ground truth vocabulary.

### Observed formats from AboutData.md

- `Art. 1 ZGB`
- `Art. 117 StGB`
- `Art. 11 Abs. 2 OR`
- `Art. 45 Abs. 2 AHVG`
- `BGE 116 Ia 56 E 1.`
- `BGE 121 III 38 E. 2b`
- `BGE 145 II 32 E. 3.1`
- `5A_800/2019 E 2.`
- `2C_123/2020 E 1.2.3`

**Key observations:**
- Article citations use period-space: `Art. 1` and `Abs. 2` (with period). Some variants have `Art.1` (no space) — these are errors relative to the corpus canonical form.
- BGE citations use Roman numerals for the division: `I`, `II`, `III`, `Ia`, `Ib`. Do NOT convert to Arabic.
- Some BGE citations have `E 1.` (no period after E) and some have `E. 2b` (period after E). **[ASSUMPTION A1]** This is inconsistent in the corpus itself — the canonicalizer should preserve whichever form the corpus row uses for that specific citation. Do NOT normalize `E.` vs `E ` away.
- Docket-format decisions use `N[A-Z]_NNN/YYYY E N.` pattern.
- `ZGB` ↔ `CC` aliasing is mentioned in D-05 but **[ASSUMPTION A2]** Swiss German corpus (laws_de.csv) will use the German-language abbreviation (ZGB) uniformly. The `CC` alias matters only if a query or LLM output produces the French form. The canonicalizer should rewrite `CC → ZGB`, not the reverse.
- `ss` vs `ß`: Swiss German uses `ss` (not `ß`). **[ASSUMPTION A3]** The corpus is Swiss, so use `ss` as canonical. Any `ß` in a query or extracted code should become `ss`.

### Canonicalization rules for Phase 1

```python
LAW_CODE_ALIASES = {
    "CC":   "ZGB",  # Code civil → Zivilgesetzbuch
    "CO":   "OR",   # Code des obligations → Obligationenrecht
    "CP":   "StGB", # Code pénal → Strafgesetzbuch
    "CPP":  "StPO", # Code de procédure pénale → Strafprozessordnung
    "LP":   "SchKG",# Loi sur la poursuite → SchKG
    "LTF":  "BGG",  # Loi sur le Tribunal fédéral → BGG
    # add more as encountered; safe to extend
}

def canonicalize(s: str) -> str:
    """
    Canonicalize a query, corpus text, or citation string to the Swiss German
    canonical form used in laws_de.csv. Idempotent.
    Rules:
      1. Collapse multiple whitespace to single space, strip.
      2. ß → ss (Swiss German standard).
      3. Normalize 'Art.X' / 'Art.  X' → 'Art. X' (single space after period).
      4. Same for 'Abs.X' → 'Abs. X', 'lit.X' → 'lit. X', 'Ziff.X' → 'Ziff. X'.
      5. Map French-language law code aliases to German (CC→ZGB etc.).
      6. Preserve BGE Roman numerals (I, II, III, Ia) — do NOT convert to digits.
      7. Preserve 'E.' and 'E ' variants — they are corpus-authentic, not errors.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("ß", "ss")
    s = re.sub(r"\s+", " ", s)
    # Normalize Art./Abs./lit./Ziff. spacing
    s = re.sub(r"\b(Art|Abs|lit|Ziff)\.\s*(\d)", r"\1. \2", s)
    # French alias rewriting — word-boundary anchored
    for fr, de in LAW_CODE_ALIASES.items():
        s = re.sub(rf"\b{fr}\b", de, s)
    return s
```

**Application points:**
1. `laws['citation'] = laws['citation'].apply(canonicalize)` before indexing (LAWS-05)
2. `laws['text'] = laws['text'].apply(canonicalize)` before indexing (LAWS-05)
3. `q_en_canon = canonicalize(row['query'])` before code extraction and encoding (QUERY-04)
4. `q_de_canon = canonicalize(translations[qid])` before BM25 tokenization (QUERY-04 + D-05)
5. `pred_cit_canon = canonicalize(cit)` before writing to submission.csv (CALIB-02)

**Round-trip invariant (assertion for the smoke test):**
```python
assert canonicalize(canonicalize("Art.11 Abs.  2 CC")) == canonicalize("Art.11 Abs.  2 CC")
assert canonicalize("Art.11 Abs.2 CC") == "Art. 11 Abs. 2 ZGB"
```

**[ASSUMPTION A4]** The exact set of aliases to include (CC/CO/CP vs more obscure ones like LAVI/LEtr/LAMal) depends on what the competition corpus actually contains. The planner should add a task: "scan `laws['citation'].unique()` for the top-100 law code suffixes and log the distribution; extend LAW_CODE_ALIASES if French codes appear." This is a 30-line data-inspection task that ships inside the notebook before the encoding stage.

---

## Swiss Citation Format — What Is / Isn't the Scorer's Vocabulary

**Canonical form source:** The exact strings in the `citation` column of `laws_de.csv` and `court_considerations.csv` are the scorer's vocabulary. `[CITED: AboutData.md §Retrieval granularity and §Can I predict citations that are not in the corpus?]`.

> "Those won't match anything and will be scored as false positives. Treat the corpus citation strings as the closed 'vocabulary' of valid outputs."

**This means:**
- The canonicalizer must produce strings that ALREADY EXIST in the corpus. A canonicalized string that doesn't match any `laws['citation']` row is worse than no prediction.
- The Phase 1 pipeline retrieves by row index and pulls `laws.iloc[idx]['citation']` — the string is corpus-authentic by construction. **No transformation is applied to the retrieved citation between retrieval and output** except for whitespace/alias canonicalization that was ALSO applied at index-build time (LAWS-05).
- If `laws['citation']` is canonicalized at load time (step 1 above), and predictions come from `laws.iloc[idx]['citation']` (also canonicalized), the pipeline is self-consistent.

**What about predictions whose canonical form collapses two corpus rows into one?** If `laws.iloc[i]['citation']` and `laws.iloc[j]['citation']` both canonicalize to the same string, that's a LAWS-05 bug — the canonicalizer is too aggressive. The smoke test should assert `len(set(canonicalize(c) for c in laws['citation'])) == len(set(laws['citation']))` at index-build time to catch this.

**[ASSUMPTION A1] (restated):** Preserving `E.` vs `E ` variants as-is means the canonicalizer is a PARTIAL normalizer — it fixes whitespace and aliases but does NOT touch the period-after-E pattern. This is the safe choice because the corpus itself is inconsistent. If the planner discovers that the competition scorer considers `BGE 116 Ia 56 E 1.` and `BGE 116 Ia 56 E. 1` as the same gold citation, this rule can be tightened. Until then: preserve.

---

## Submission Validator (CALIB-03 / MP-5)

```python
def validate_submission(df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Assert the submission DataFrame matches competition format.
    Fail loudly — never write a bad CSV.
    """
    # Columns
    assert list(df.columns) == ["query_id", "predicted_citations"], (
        f"wrong columns: {list(df.columns)}"
    )
    # Row count matches test set
    assert len(df) == len(test_df), (
        f"row count mismatch: submission={len(df)} test={len(test_df)}"
    )
    # All test query_ids present, no duplicates
    assert set(df["query_id"]) == set(test_df["query_id"]), (
        "query_id set does not match test.csv"
    )
    assert df["query_id"].is_unique, "duplicate query_id in submission"
    # No NaN
    assert not df["predicted_citations"].isna().any(), "NaN in predicted_citations"
    # Values are strings (may be empty per AboutData.md: "empty string allowed")
    for i, val in df["predicted_citations"].items():
        assert isinstance(val, str), f"row {i}: not a string"
        # No trailing semicolons
        assert not val.endswith(";"), f"row {i}: trailing semicolon"
        # No commas inside citation strings (would break CSV under escaping)
        # Note: commas ARE legal inside the quoted CSV field per AboutData.md example
        #       ("Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1") but a comma inside a single
        #       citation is suspicious. Warn, don't assert.
        for cit in val.split(";"):
            if cit and "," in cit:
                print(f"  WARN row {i}: comma inside citation '{cit}'", flush=True)
    print(f"validate_submission OK: {len(df)} rows", flush=True)
```

Called as:
```python
validate_submission(submission_df, test)
submission_df.to_csv(OUT_PATH, index=False)
```

**Source:** The format spec at `[CITED: AboutData.md §Submission format]` — `query_id,predicted_citations` with semicolons and empty strings allowed.

---

## FAISS Index Strategy

**Laws corpus: `IndexFlatIP` with 1024-dim float32, 175K docs.**

Size: `175_000 × 1024 × 4 bytes = 717 MB`. Fits trivially in T4 VRAM OR in 30GB RAM.

```python
import faiss

if USE_FAISS_GPU:
    # GPU path — slightly faster but not required
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(1024)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
else:
    # CPU path — works on local Windows and as Kaggle fallback
    index = faiss.IndexFlatIP(1024)

index.add(laws_embs)  # laws_embs is (175K, 1024) L2-normalized float32
print(f"FAISS: {index.ntotal:,} vectors, d={index.d}")

# Query
q_vec = dense_query_embedding(q_en_canon, q_de_canon)  # (1, 1024)
D, I = index.search(q_vec, DENSE_LAWS_K)
dense_ids = I[0].tolist()
```

**Why IndexFlatIP not IVF/HNSW:**
- 175K docs is small enough that brute-force exact search costs <1 ms per query on GPU, ~5 ms on CPU. IVF/HNSW add approximation error for zero benefit.
- IndexFlatIP is deterministic (MP-2: HNSW is non-deterministic).
- IP on L2-normalized vectors = cosine similarity, which is what BGE-M3 is trained for.

**Why NOT the GPU index on Windows local dev:** `faiss-gpu-cu12` is Linux x86_64 only `[VERIFIED: pypi.org/project/faiss-gpu-cu12/]`. The `IS_KAGGLE` switch already gates on platform; extend it to gate on `USE_FAISS_GPU`.

---

## OpusMT Loading (QUERY-01 / D-01)

**Model:** `Helsinki-NLP/opus-mt-tc-big-en-de`, MarianMT family, ~1.2 GB disk / ~298M params on disk `[CITED: .planning/research/STACK.md; huggingface.co/docs/transformers/model_doc/marian]`.

```python
# Cell 4 — OpusMT load, translate all queries, unload
from transformers import AutoTokenizer, MarianMTModel

OPUSMT_DIR = Path("/kaggle/input/opus-mt-tc-big-en-de") if IS_KAGGLE else "Helsinki-NLP/opus-mt-tc-big-en-de"

print(f"Loading OpusMT from {OPUSMT_DIR}")
t0 = time.time()
opus_tok = AutoTokenizer.from_pretrained(str(OPUSMT_DIR))
opus_mdl = MarianMTModel.from_pretrained(str(OPUSMT_DIR)).to(DEVICE).eval()
print(f"  loaded in {time.time()-t0:.1f}s; device={next(opus_mdl.parameters()).device}")

@torch.no_grad()
def translate_en_de(texts, batch_size=8, max_len=512):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = opus_tok(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len).to(DEVICE)
        gen = opus_mdl.generate(**enc, max_new_tokens=max_len, num_beams=4)
        out.extend([opus_tok.decode(g, skip_special_tokens=True) for g in gen])
    return out

# Translate all val + test queries in one batch
all_queries_en = val["query"].tolist() + test["query"].tolist()
all_query_ids  = val["query_id"].tolist() + test["query_id"].tolist()

t0 = time.time()
all_queries_de = translate_en_de(all_queries_en, batch_size=8)
print(f"Translated {len(all_queries_en)} queries in {time.time()-t0:.1f}s")

translations = dict(zip(all_query_ids, all_queries_de))  # qid → German text

# D-02: unload before BGE-M3 loads
unload(opus_tok, opus_mdl)
```

**Batch size:** The existing `solution.py:180` uses `batch_size=4`. On T4 with a 1.2GB model and ~50 queries, `batch_size=8` fits comfortably with `num_beams=4`. `num_beams=4` improves legal-domain translation quality modestly; `num_beams=1` (greedy) is faster and fine for Phase 1 if VRAM pressure is a concern.

**[ASSUMPTION A6]** Expected translation time on T4: ~5-15 seconds total for 50 queries at `batch_size=8, num_beams=4`. Not verified against the specific tc-big variant on T4 — flag for smoke test. If translation exceeds 60 seconds, drop to `num_beams=1`.

**Model download for Kaggle dataset packaging:** The existing `download_models.py` maps `Helsinki-NLP/opus-mt-en-de → models/opus-mt-en-de/`. Phase 1 must either:
1. Add `Helsinki-NLP/opus-mt-tc-big-en-de → models/opus-mt-tc-big-en-de/` to `download_models.py` and upload as a new Kaggle dataset, OR
2. Keep using the smaller `opus-mt-en-de` for Phase 1 and upgrade to `tc-big` in Phase 2.

**Recommendation:** Option 1 — pay the ~1GB upload cost now, avoid churning models between phases.

---

## Cross-Encoder Reranking

**Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — already in `models/`, already in `solution.py:194-214`. Phase 1 reuses unchanged except for the sequential lifecycle (load AFTER BGE-M3 is unloaded).

```python
# Cell 10 — Rerank top-RERANK_K candidates per query
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RERANK_DIR = Path("/kaggle/input/mmarco-reranker") if IS_KAGGLE else "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
rerank_tok = AutoTokenizer.from_pretrained(str(RERANK_DIR))
rerank_mdl = AutoModelForSequenceClassification.from_pretrained(str(RERANK_DIR)).to(DEVICE).eval()
print(f"Reranker loaded: device={next(rerank_mdl.parameters()).device}")

@torch.no_grad()
def rerank(query_en, candidate_texts, batch_size=32):
    scores = []
    for i in range(0, len(candidate_texts), batch_size):
        batch_txts = candidate_texts[i:i+batch_size]
        enc = rerank_tok(
            [query_en] * len(batch_txts), batch_txts,
            return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(DEVICE)
        out = rerank_mdl(**enc)
        scores.extend(out.logits.squeeze(-1).cpu().float().numpy().tolist())
    return np.array(scores)
```

**Pool size & batch:** `RERANK_K=150` candidates × `batch_size=32` = 5 forward passes per query × 50 queries = 250 forward passes total. At ~100-200ms per forward pass on T4, this is **30-90 seconds total** — negligible in the runtime budget. `[CITED: .planning/research/PITFALLS.md §CP-8 for batch size justification]`.

**Query language for reranker:** Pass the **English original query** (`q_en`) to the cross-encoder, not the German translation. mmarco is trained primarily on MS-MARCO English passages but supports cross-lingual inputs. `solution.py:377` already does this. `[VERIFIED: solution.py]`

**Why not jina v2?** FUSE-03 assigns jina to Phase 4. Phase 1 scope locks on mmarco.

---

## Calibration (CALIB-01 / CALIB-04 / CALIB-05)

Reuse `calibrate_top_k()` from `solution.py:303-313` unchanged except:
- Expand K search range to [1, 60] to accommodate the val mean of ~25 citations.
- Log warning if `best_k ∈ {1, 59}` (boundary hit → overfitting signal per CP-4).
- Bootstrap the val F1 confidence interval: run 10 resamples of val, report mean ± std. If std > 0.05, the K estimate is unstable and we should NOT commit to it as a test strategy.

```python
def calibrate_top_k(val_results, k_min=1, k_max=60):
    """Grid search for best fixed top-k that maximizes val macro-F1."""
    best_f1, best_k = 0.0, 20
    gold_sets = [r["gold"] for r in val_results]
    for k in range(k_min, k_max + 1):
        pred_sets = [set(r["ranked_citations"][:k]) for r in val_results]
        f1 = macro_f1(gold_sets, pred_sets)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    # Boundary warning
    if best_k == k_min or best_k == k_max:
        print(f"  WARN: best_k={best_k} at boundary of [{k_min}, {k_max}]; overfitting risk (CP-4)", flush=True)
    print(f"*** Val macro-F1 = {best_f1:.4f}  @  top-{best_k} ***")
    return best_k, best_f1
```

**Reporting format:** Match the existing `solution.py:312` banner exactly so logs diff cleanly across phases.

**Do NOT:**
- Calibrate on train set (AH-5: train mean 4.1 ≠ val mean 25.1).
- Use a single fixed K without calibration (AH-3: fixed 100 produces near-zero precision).
- Skip the bootstrap confidence interval — it's cheap and it catches the case where val F1 jumped because of a single lucky query (CP-4).

---

## Smoke Test Architecture (FOUND-05)

**Budget:** 5 minutes end-to-end on 3 val queries.

```python
# Cell 0 or top of cell 5
SMOKE = os.environ.get("SMOKE", "0") == "1"  # controllable via env

# Cell 6: laws encoding — truncate to 5000 docs in smoke mode
SMOKE_LAWS_N = 5000
if SMOKE:
    print(f"SMOKE mode: truncating laws corpus to {SMOKE_LAWS_N} docs")
    laws = laws.iloc[:SMOKE_LAWS_N].reset_index(drop=True)

# Cell 7+11: val processing — first 3 queries in smoke mode
SMOKE_VAL_N = 3
val_for_run = val.iloc[:SMOKE_VAL_N] if SMOKE else val
```

Smoke-mode expected wall-clock:
- CUDA assert: <1s
- OpusMT translate: ~5-10s (still on full val+test count, 50 queries — cheap)
- BGE-M3 load: ~15s
- Laws encoding (5000 docs, batch 64): ~30-60s
- bm25s build (5000 docs): ~2s
- BGE-M3 query encode + FAISS kNN × 3 queries: ~1s
- bm25s kNN × 3 queries: ~0.1s
- RRF fusion: <0.1s
- Cross-encoder load: ~10s
- Rerank (150 cands × 3 queries): ~5s
- Calibrate on 3 queries + validate + write: <1s

**Total: ~75-105 seconds.** Comfortably under the 5-minute FOUND-05 budget.

**Smoke-mode pass criteria:**
- No exception
- `validate_submission(...)` passes
- Val F1 on 3 queries is logged (value can be 0.0 — the point is that the pipeline runs end-to-end without errors, not that it scores well on 3 queries)

---

## Model Download / Kaggle Dataset Packaging

`download_models.py` must be updated for Phase 1:

```python
MODELS = {
    "intfloat/multilingual-e5-large":              "models/multilingual-e5-large",       # keep for fallback
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1":  "models/mmarco-reranker",
    "Helsinki-NLP/opus-mt-en-de":                  "models/opus-mt-en-de",               # keep for fallback
    "Helsinki-NLP/opus-mt-tc-big-en-de":           "models/opus-mt-tc-big-en-de",        # NEW for Phase 1
    "BAAI/bge-m3":                                 "models/bge-m3",                       # NEW for Phase 1
}
```

**BGE-M3 download size:** ~2.3 GB on disk (fp32 model; fp16 cast is done at load time in PyTorch).

**ALLOW pattern must include:** `tokenizer.json`, `sentencepiece.bpe.model` (BGE-M3 uses sentencepiece), `config.json`, `model.safetensors`, `pytorch_model.bin`, `special_tokens_map.json`, `tokenizer_config.json`. The existing `ALLOW` list at `download_models.py:10-23` already covers these.

**Kaggle dataset upload:**
- `bge-m3` — new dataset, slug `bge-m3`, ~2.3 GB
- `opus-mt-tc-big-en-de` — new dataset, slug `opus-mt-tc-big-en-de`, ~1.2 GB

**Kaggle path constants** (extend `solution.py:20-35`):
```python
if IS_KAGGLE:
    DATA_DIR        = Path("/kaggle/input/llm-agentic-legal-information-retrieval")
    BGE_M3_DIR      = Path("/kaggle/input/bge-m3")                   # NEW
    OPUS_MT_DIR     = Path("/kaggle/input/opus-mt-tc-big-en-de")    # NEW (tc-big variant)
    RERANK_DIR      = Path("/kaggle/input/mmarco-reranker")
    OUT_PATH        = Path("/kaggle/working/submission.csv")        # fix: not ./ on Kaggle
else:
    DATA_DIR        = Path(r"C:\Users\Dharun prasanth\OneDrive\Documents\Projects\LLm_Agentic\Data")
    BGE_M3_DIR      = "BAAI/bge-m3"
    OPUS_MT_DIR     = "Helsinki-NLP/opus-mt-tc-big-en-de"
    RERANK_DIR      = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    OUT_PATH        = Path(r"C:\Users\Dharun prasanth\OneDrive\Documents\Projects\LLm_Agentic\submission.csv")
```

**Note:** `solution.py:29` currently writes to `Path("submission.csv")` on Kaggle — a relative path. The Kaggle working directory is `/kaggle/working/`, so this happens to work, but the explicit absolute path is safer.

---

## Runtime Budget (FOUND-06)

Phase 1 full-run wall-clock estimate on Kaggle T4:

| Stage | Expected | Notes |
|-------|----------|-------|
| Pip installs (Kaggle cell 0) | 30-60s | bm25s, faiss-gpu-cu12, CharSplit, nltk — small packages |
| CUDA assert + imports (cell 1) | 2s | |
| Load CSVs (cell 2) | 20-40s | 175K laws + 50 queries; court corpus NOT loaded in Phase 1 |
| Canonicalize helpers defined (cell 3) | 0s | |
| OpusMT translate 50 queries (cell 4) | 10-20s | Load + translate + unload |
| BGE-M3 load fp16 (cell 5) | 15-30s | First-time load from disk |
| BGE-M3 encode 175K laws (cell 6) | 4-8 min | 64 batch × ~30 batches/sec on T4 |
| FAISS IndexFlatIP build (cell 6 tail) | 5-10s | |
| BGE-M3 encode 50 queries ×2 (EN+DE) (cell 7) | 2-5s | |
| FAISS kNN × 50 queries (cell 7) | <1s | |
| Unload BGE-M3 (cell 7 tail) | 2s | |
| bm25s laws index build (cell 8) | 1-3 min | Includes CharSplit tokenization of 175K docs |
| bm25s kNN × 50 queries (cell 8) | <1s | |
| RRF fuse (cell 9) | <1s | |
| Cross-encoder load (cell 10) | 5-10s | |
| Rerank 150 cands × 50 queries (cell 10) | 1-3 min | `[CITED: CP-8 cross-encoder batch math]` |
| Unload cross-encoder (cell 10 tail) | 2s | |
| Calibrate K on val (cell 11) | <1s | Grid search over 60 values × 10 val queries |
| Apply best_k to test, canonicalize, validate, write (cell 12) | <1s | |
| **Total expected** | **12-18 min** | |

**Safety margin against 11-hour budget:** ~10+ hours. Phase 1 is far from the runtime ceiling. The 11-hour target matters for Phase 2 (2.47M court BM25 index) and Phase 5 (Qwen LLM stage), not Phase 1.

**[ASSUMPTION A7]** The 4-8 minute laws encoding estimate is based on `.planning/research/STACK.md` stating "~500 docs/sec on T4" for BGE-M3-class models. The actual rate depends on max_length, fp16 vs fp32, and Kaggle kernel tail latency. Smoke test on 5000 docs will empirically verify before committing to the full run.

---

## Reproducibility (INC-01 / INC-02 / INC-03)

### Seeds

Set at cell 1, covering all sources of randomness:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

**What's still non-deterministic:** cross-encoder scoring is deterministic under `.eval()` + `no_grad()`; BGE-M3 likewise. FAISS `IndexFlatIP` is exact, therefore deterministic. bm25s scoring is deterministic.

**Sources of possible non-determinism NOT covered by seeds:**
- cuBLAS non-determinism for matmul (minor; set `torch.use_deterministic_algorithms(True)` if it bites, but this breaks some ops — not recommended for Phase 1)
- Kaggle kernel tail memory state affecting the exact sequence of `torch.cuda.empty_cache()` outcomes (affects timing, not outputs)

### Feature flags (INC-02)

All phases share these flags, defined at top of cell 0 / 1:

```python
USE_RERANKER          = True   # Phase 1: on
USE_COURT_CORPUS      = False  # Phase 2+ enables
USE_ENTITY_GRAPH      = False  # Phase 3+ enables
USE_COURT_DENSE_RERANK = False # Phase 4+ enables
USE_JINA_CROSS_ENCODER = False # Phase 4+ replaces mmarco
USE_LLM_AUGMENTATION  = False  # Phase 5+ enables
SMOKE                 = False  # Test mode: laws[:5000], val[:3]
```

Future phases flip the flags ON without editing existing logic.

### Git tags (INC-03)

After `validate_submission()` passes and `submission.csv` is written, the final plan task should be:

```bash
git add submission.csv notebook_kaggle.ipynb solution.py download_models.py requirements.txt \
        .planning/phases/01-foundation-laws-pipeline/01-PLAN.md \
        .planning/phases/01-foundation-laws-pipeline/01-VERIFY.md
git commit -m "phase 1: foundation + laws pipeline — val F1 <X>"
git tag phase-1-submission
```

**Remember:** Per parent CLAUDE.md, commit messages have **no** "Claude" / "Anthropic" / "Co-Authored-By" attribution. The planner must produce commit messages that look fully human-authored.

**Push is user-gated.** Per GSD convention and CLAUDE.md, the agent does not `git push` without explicit user request.

---

## Runtime State Inventory

Phase 1 is a **stack swap + new-code** phase, not a rename/refactor phase. The only runtime state that matters:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — laws/court CSVs are read-only input. The pipeline writes only `submission.csv`. | None |
| Live service config | None — Kaggle notebook is the only service; no separate config | None |
| OS-registered state | None | None |
| Secrets/env vars | `kaggle.json` contains Kaggle API credentials (already present at repo root per STRUCTURE.md). Not modified by Phase 1. | None |
| Build artifacts | `models/` cache is local to dev machine; Kaggle datasets are separate uploads. `download_models.py` must add 2 new models. | Update `download_models.py` + upload 2 new Kaggle datasets |

**Nothing accumulates across notebook runs except `submission.csv`**, which is overwritten each run by design.

---

## Common Pitfalls

### Pitfall 1: BGE-M3 mean-pooled by accident
**What goes wrong:** Copy-paste from `solution.py` brings `mean_pool()` into the BGE-M3 encode path. Embeddings are wrong but FAISS still returns results — recall silently degraded by 10-30%.
**Why it happens:** `mean_pool` was correct for e5-large; naming it `mean_pool` and reusing the helper in a new encoder is tempting.
**How to avoid:** Write `bge_encode()` as a new function with CLS pooling inline. Do NOT import or reuse `mean_pool()`. Smoke-test assertion: `bge_encode(["Art. 1 ZGB"])` produces a non-zero variance embedding (mean_pool on a single long-token string can produce near-zero variance, which is a tell).
**Warning signs:** Dense retrieval recall@100 < 10% on val; top-FAISS-hits are unrelated to the query; BM25 dominates RRF.

### Pitfall 2: e5 prefixes added to BGE-M3
**What goes wrong:** `encode()` from `solution.py:133` has `prefix="passage: "` default. If the planner ports this to `bge_encode()`, every embedding gets poisoned by a leading token that BGE-M3 was not trained on.
**Why it happens:** Prefix parameter is a sticky API detail; the function signature looks generic.
**How to avoid:** No `prefix` parameter in `bge_encode()`. Pass raw text. Smoke-test: `bge_encode(["Art. 1 ZGB"])` and `bge_encode(["query: Art. 1 ZGB"])` should produce different embeddings (they will for BGE-M3 — and the bare one is correct).
**Warning signs:** Adding the same passage as both "query" and "passage" variants produces >0.999 cosine similarity (means the prefix doesn't matter to the model = e5; different similarities = BGE-M3).

### Pitfall 3: `faiss-gpu-cu12` install failure on Kaggle kernel
**What goes wrong:** Kaggle kernels have specific CUDA runtime versions. `faiss-gpu-cu12[fix-cuda]` requires NVIDIA Driver ≥R530, which Kaggle T4 kernels generally have, but the `[fix-cuda]` extra may conflict with Kaggle's shipped CUDA libs.
**Why it happens:** pip wheel picks up incompatible cuBLAS via PyPI.
**How to avoid:** Use `pip install faiss-gpu-cu12` WITHOUT `[fix-cuda]` (lets it pick up Kaggle's CUDA). Catch `subprocess.CalledProcessError` and fall back to `faiss-cpu`. `faiss-cpu` is already in requirements.txt.
**Warning signs:** `ImportError: libcudart.so.12: cannot open shared object file`; import succeeds but `faiss.StandardGpuResources()` crashes.

### Pitfall 4: OpusMT tc-big eats too much VRAM in fp32
**What goes wrong:** Loading `opus-mt-tc-big-en-de` in fp32 takes ~1.2GB alone; with beam-4 generation on 8-query batches the activation memory doubles. At 2-3GB peak, this still fits on T4 but leaves less headroom for BGE-M3 load if the unload is incomplete.
**Why it happens:** `MarianMTModel.from_pretrained()` defaults to fp32.
**How to avoid:** Load in fp16 via `.half()` OR keep fp32 but ensure `del opus_mdl; gc.collect(); torch.cuda.empty_cache()` actually runs before BGE-M3 loads. Log VRAM free before and after unload to catch incomplete unloads.
**Warning signs:** `CUDA out of memory` when BGE-M3 loads; `torch.cuda.mem_get_info()[0]` after OpusMT unload is suspiciously low.

### Pitfall 5: bm25s tokenization treats 'Art.' as stopword or drops the period
**What goes wrong:** `tokenize_for_bm25_de()` uses `re.findall(r"[\w.]+")` which keeps periods, but NLTK's German stopword list does not contain "art" — good. However, CharSplit might see `"art."` as a non-compound, return no split, and the legal-notation dots could be stripped by a greedy `\w` regex pass later.
**Why it happens:** The interaction between the regex splitter and CharSplit is subtle.
**How to avoid:** Assert after tokenizing a smoke-test query: `"Art." in tokenize_for_bm25_de("Die Pflicht aus Art. 41 OR")` — the legal code must survive.
**Warning signs:** BM25 retrieval returns 0 results for queries that explicitly mention `Art. X OR`.

### Pitfall 6: Canonicalizer collapses two distinct corpus citations
**What goes wrong:** `canonicalize("Art.11 Abs. 2 CC") == canonicalize("Art. 11 Abs. 2 CC")` is intentional. But if the corpus has BOTH `Art. 11 Abs. 2 OR` and `Art. 11 Abs. 2 CO` (legitimate different abbreviations), the CO→OR alias rewrite collapses them. This is wrong if they point to different laws.
**Why it happens:** Alias table is too aggressive.
**How to avoid:** The smoke test described in §Swiss Citation Canonicalization — `len(set(canonicalize(c) for c in laws['citation'])) == len(set(laws['citation']))`. If this fails, shrink the alias table to only aliases that genuinely point to the same law.
**Warning signs:** Smoke-test assertion fails.

### Pitfall 7: Calibration best_k lands on boundary
**What goes wrong:** `calibrate_top_k(val_results, k_max=60)` returns `best_k=60`. This means the grid search wanted to go higher but hit the cap.
**Why it happens:** Val F1 is monotonically increasing with K up to 60 (pure recall, no precision cost) — indicates the retrieval is finding gold citations but ranking them low.
**How to avoid:** Print the warning, then inspect the per-query F1 at K=60. If all queries have F1 > 0 at K=60, widen the cap to 100 and re-calibrate. If several queries have F1 = 0 even at K=60, the retrieval itself is broken — investigate dense/sparse recall@K separately.
**Warning signs:** `best_k == k_max`; CP-4 warning.

---

## Code Examples

Verified patterns (sources cited inline):

### BGE-M3 dense encode (CLS pool + L2 normalize, no prefixes)

```python
# Source: [CITED: huggingface.co/BAAI/bge-m3/discussions/17, /35, /80]
@torch.no_grad()
def bge_encode(texts, batch_size=64, max_length=512):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t)[:2000] for t in texts[i:i+batch_size]]
        enc = bge_tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(DEVICE)
        out = bge_mdl(**enc)
        emb = out.last_hidden_state[:, 0]                  # CLS token
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2 normalize
        all_embs.append(emb.float().cpu().numpy())
    return np.vstack(all_embs).astype("float32")
```

### FAISS GPU/CPU conditional build

```python
# Source: [CITED: .planning/research/STACK.md; pypi.org/project/faiss-gpu-cu12/]
import faiss

cpu_index = faiss.IndexFlatIP(1024)
if USE_FAISS_GPU:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
else:
    index = cpu_index
index.add(laws_embs)
D, I = index.search(q_vec, k=DENSE_LAWS_K)
```

### bm25s with pre-tokenized input

```python
# Source: [CITED: github.com/xhluca/bm25s]
import bm25s

# list-of-list tokens accepted directly
tokens = [tokenize_for_bm25_de(t) for t in laws_texts]
retriever = bm25s.BM25()
retriever.index(tokens)

# Query
q_tokens = tokenize_for_bm25_de(q_de_canonical + " " + legal_codes)
results, scores = retriever.retrieve([q_tokens], k=BM25_LAWS_K)
bm25_ids = results[0].tolist()
```

### RRF fusion (reuse existing)

```python
# Source: [VERIFIED: solution.py:220-226]
def rrf_fuse(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Phase 1 use: 2 ranked lists
fused = rrf_fuse([dense_laws_ids, bm25_laws_ids])
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `multilingual-e5-large` with `"query: "`/`"passage: "` prefixes and mean pooling | `BAAI/bge-m3` with no prefixes and CLS pooling | BGE-M3 released 2024-Q1 | Cross-lingual MIRACL nDCG@10 ~65.4 → ~67.8; 8192 token context vs 512 |
| `multilingual-e5-small` (117M, 384-dim) | BGE-M3 (560M, 1024-dim) | Same change | Recall jumped from near-zero to meaningful (AH-4 team-confirmed) |
| `rank-bm25` pure-Python loops | `bm25s` scipy sparse + Numba | bm25s released 2024-07 | 500× faster; enables 2.47M court corpus indexing for Phase 2 |
| `faiss-gpu` (deprecated wheel) | `faiss-gpu-cu12` | faiss-gpu deprecated 1.7.3 | Working pip install; CUDA 12 compatible |
| `Helsinki-NLP/opus-mt-en-de` | `Helsinki-NLP/opus-mt-tc-big-en-de` | tc-big variant added 2022 | Noticeably better German legal translation; 4× model size still fits |
| Fixed K=100 predictions per query | Per-query K calibrated on val | AH-3 team-confirmed failure | Macro-F1 precision restored |

**Deprecated/outdated:**
- `pip install faiss-gpu` — the wheel is dead. Use `faiss-gpu-cu12` (Linux) or `faiss-cpu` (everything).
- `rank_bm25.BM25Okapi` for anything larger than 10K docs — use `bm25s.BM25` instead.
- `multilingual-e5-small` for cross-lingual retrieval — too small for legal domain; use e5-large or BGE-M3.
- Forcing `DEVICE = torch.device('cpu')` in `notebook_kaggle.ipynb` cell 1 — explicit AH-1 bug; the new notebook asserts CUDA.

---

## Assumptions Log

Claims tagged `[ASSUMED]` or `[ASSUMPTION A#]` in this research that the planner and/or executor must confirm:

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `BGE 116 Ia 56 E 1.` and `BGE 116 Ia 56 E. 1` should be preserved as-is (NOT normalized) because the corpus itself is inconsistent | §Swiss Citation Canonicalization | If the scorer treats them as distinct and our canonicalizer leaves them alone, we're safe. If the scorer collapses them AND our canonicalizer also collapses them to the corpus form, we're safe. Risk: if the scorer collapses them but we leave both forms, we emit duplicate citations in submission.csv. **Mitigation:** plan a data-inspection task that counts distinct `E.` vs `E ` variants in the corpus and reports the ratio. |
| A2 | Swiss German corpus uses German abbreviations (ZGB, OR, StGB) uniformly; the French aliases (CC, CO, CP) appear only in queries or LLM output | §Swiss Citation Canonicalization | If laws_de.csv actually contains both ZGB and CC entries for the Civil Code, aliasing CC→ZGB during canonicalization collapses two distinct rows. **Mitigation:** data-inspection task — run `laws['citation'].str.extract(r'(ZGB|CC|OR|CO|StGB|CP)').value_counts()` before applying the alias rewrite. |
| A3 | `ss` is the canonical Swiss German form; `ß` should be rewritten to `ss` in queries and output | §Swiss Citation Canonicalization | If any corpus entries actually use `ß` (unlikely for Swiss but not impossible), the rewrite is still harmless because no gold citation would contain `ß` either. Low risk. |
| A4 | The alias table should include CC/CO/CP/CPP/LP/LTF; other French codes (LAVI/LEtr/LAMal) are unlikely to appear in English queries | §Swiss Citation Canonicalization | If English queries use Swiss French abbreviations we didn't alias, BM25 on the translated German query misses them. **Mitigation:** the data-inspection task should also scan `val['query']` and `test['query']` for ALL-CAPS tokens of length 2-6 and cross-reference. |
| A5 | `CharSplit` PyPI package name is stable and the Splitter class API is `splitter.split_compound(word) -> list[(score, head, tail)]` | §bm25s Integration | Wrong install name or API breaks the laws BM25 tokenization cell. **Mitigation:** plan a task "install CharSplit; if that fails, install `german_compound_splitter`; smoke-test both API shapes." |
| A6 | OpusMT tc-big translation of 50 queries takes 5-15s on T4 with beam=4 | §OpusMT Loading | If it takes much longer (e.g., 60s+), it eats into the smoke test budget. **Mitigation:** smoke test flags it; fall back to `num_beams=1`. |
| A7 | BGE-M3 encoding rate on T4 is ~500 docs/sec → 175K docs in 4-8 min | §Runtime Budget | If it's 10× slower than expected (e.g., 50 docs/sec), the full run takes 1 hour just for laws encoding, still fine for 11h budget but eats the "smoke test completes in 5 min" constraint for anything beyond 5K docs. **Mitigation:** the 60-second checkpoint catches an order-of-magnitude slowdown; measured rate is logged. |
| A8 | Kaggle T4 kernels ship a torch version that supports fp16 BGE-M3 load via `torch_dtype=torch.float16` | §Pattern 2 | If the kernel ships a torch too old (<2.0) to handle `torch_dtype` arg, the fp16 load syntax breaks. **Mitigation:** smoke-test on day-1 of plan execution; if it breaks, use `.half()` after load. |
| A9 | `download_models.py`'s `ALLOW` pattern list is sufficient for BGE-M3 (it needs `tokenizer.json`, `sentencepiece.bpe.model`, `config.json`, `model.safetensors`) | §Model Download / Kaggle Dataset Packaging | If BGE-M3's repo has a file not matched by ALLOW, the download is incomplete and `AutoModel.from_pretrained` fails at load time with a cryptic message. **Mitigation:** verify with `ls models/bge-m3/` after download; should contain all 4+ files. |
| A10 | The 10-query val set's gold citation distribution matches the 40-query test distribution closely enough that `best_k` calibrated on val will transfer | §Calibration | AboutData.md says "It matches the val distribution" for test `[CITED: AboutData.md]`, so this is explicitly supported — low risk. |

**How to read this table:** Every `[ASSUMED]` or `[ASSUMPTION A#]` tag in this document corresponds to a row here. When the executor runs the plan, the first few cells should be a "data inspection" batch that converts A1-A5 from assumptions into verified facts. If any inspection contradicts the assumption, the executor stops and asks the user (or re-plans).

---

## Open Questions

1. **Which BGE-M3 `max_length` is optimal for Swiss legal text?**
   - What we know: BGE-M3 supports 8192 tokens `[CITED: huggingface.co/BAAI/bge-m3]`. `solution.py` uses `max_length=512` for e5-large (e5's hard limit).
   - What's unclear: Swiss law snippets in `laws_de.csv` can be long (full article text). At `max_length=512`, we truncate ~20-40% of long rows. At `max_length=1024`, encoding time doubles. At `max_length=2048`, it quadruples.
   - Recommendation: **Start with `max_length=512` for Phase 1** — matches current code, cheap, and the `citation + first_400_chars` snippet pattern at `solution.py:158-161` already pre-truncates to a manageable size. Revisit in Phase 4 if dense recall is a bottleneck.

2. **Should the dense query embedding average [EN, DE] equally (D-04) or weight one?**
   - What we know: D-04 says "averages the two embeddings L2-normalized" — equal weight is the locked decision.
   - What's unclear: Whether equal weight is actually optimal. BGE-M3 handles EN queries natively; the German translation may add noise for some queries.
   - Recommendation: **Honor D-04 (equal weight) in Phase 1.** Log per-query dense retrieval recall separately for {EN-only, DE-only, averaged} on val as a diagnostic — this surfaces the question for later phases without relitigating D-04.

3. **Does the cross-encoder need truncation at 256 tokens or 512 tokens per candidate?**
   - What we know: CP-8 (`.planning/research/PITFALLS.md`) recommends 256-token truncation at 200 candidates for court corpus. Phase 1 has no court corpus and only 150 candidates.
   - What's unclear: Whether 512-token truncation is affordable at 150 candidates.
   - Recommendation: **Use 512 tokens at 150 candidates for Phase 1.** Total forward passes = 150 × 50 / 32 = ~235. At 200ms each = 47 seconds. Fits the budget with room. Phase 4 tightens this when the fused candidate pool grows.

4. **bm25s `stopwords` string vs pre-filtered list — which works reliably?**
   - What we know: bm25s's `stopwords="de"` uses an ISO code mapping that `[ASSUMED]` may map to NLTK's "german" list.
   - What's unclear: Whether the ISO mapping in bm25s 0.2.x is robust or silently returns an empty list.
   - Recommendation: **Pre-filter in our own tokenizer**, pass clean list-of-list to `bm25s.BM25().index()`. Avoids the ambiguity entirely.

5. **What's the actual val gold-citation count distribution, and does it shift K calibration?**
   - What we know: AH-5 says val mean is 25.1; train mean is 4.1. AboutData.md says test "matches the val distribution."
   - What's unclear: Whether val's distribution is [20, 22, 25, 28, 31] (tight) or [2, 10, 25, 50, 100] (wide). Wide distribution means a single global K is fundamentally wrong — per-query K is needed (Phase 5 scope).
   - Recommendation: **Add a data-inspection task to cell 2**: log `val['gold_citations'].str.split(';').str.len().describe()`. If std > 10, flag the result in the calibration cell. Do NOT act on it in Phase 1 — per-query K is locked to Phase 5.

---

## Environment Availability

Phase 1 requires the following external tools and services. Verification is partial because the working directory is Windows local dev and the actual runtime is Kaggle kernel:

| Dependency | Required By | Available (local dev) | Available (Kaggle) | Version | Fallback |
|------------|------------|----------------------|--------------------|---------|----------|
| Python 3.10-3.13 | All code | ✓ | ✓ | 3.10+ Kaggle | — |
| torch 2.1+ | All model inference | ✓ (verified: requirements.txt) | ✓ (Kaggle kernel default) | 2.1+ | — |
| transformers 4.38+ | BGE-M3, OpusMT, mmarco | ✓ | ✓ | 4.38+ | — |
| bm25s 0.2.x | Laws BM25 index | ✗ (not in requirements.txt) | ✗ (pip install in cell 0) | 0.2.0+ | Add to requirements.txt + `pip install bm25s` |
| faiss-gpu-cu12 | GPU FAISS | ✗ Linux-only; Windows is faiss-cpu | **UNVERIFIED** — needs Kaggle kernel test | 1.14.1+ | `faiss-cpu` (already in requirements.txt) |
| faiss-cpu | CPU FAISS (fallback) | ✓ (requirements.txt) | ✓ | 1.7.4+ | — |
| CharSplit or german_compound_splitter | German decompounding | ✗ | ✗ (pip install cell 0) | — | Pure-regex tokenization without decompounding (degrades recall, not correctness) |
| NLTK german stopwords | bm25s stopword filter | ✗ | ✗ (pip install + nltk.download) | 3.8+ | Hardcoded stopword list inline |
| BAAI/bge-m3 model weights | Dense encoding | ✓ via `huggingface-hub` download | ✗ — must be uploaded as Kaggle dataset | — | Cannot fall back; required for LAWS-01 |
| Helsinki-NLP/opus-mt-tc-big-en-de | Translation | ✓ via `huggingface-hub` download | ✗ — must be uploaded as Kaggle dataset | — | Fall back to `Helsinki-NLP/opus-mt-en-de` (already in models/) |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Reranker | ✓ (already in models/) | ✓ (already a Kaggle dataset) | — | — |
| Kaggle T4 GPU | All inference | — | **UNVERIFIED at plan time** — user must confirm notebook accelerator setting is T4 ×2 | CC 7.5 | Hard-fails at cell 1 (D-07) |

**Missing dependencies with no fallback:**
- **BGE-M3 model weights uploaded as a Kaggle dataset.** This is a hard prerequisite for running the notebook on Kaggle. The plan must include a task: "(1) run `download_models.py` locally to fetch BGE-M3, (2) upload `models/bge-m3/` as a new Kaggle dataset named `bge-m3`, (3) attach the dataset to the notebook."

**Missing dependencies with fallback:**
- **`faiss-gpu-cu12` on Kaggle** — detected at import time; falls back to `faiss-cpu` which is already in requirements.txt and works for 175K IndexFlatIP with negligible latency cost.
- **`opus-mt-tc-big-en-de`** — if the upload fails or is delayed, fall back to `opus-mt-en-de` (already in models/, ~300MB smaller, slightly lower translation quality). The fallback does NOT block Phase 1.
- **CharSplit** — if the package install fails, the tokenizer just emits the raw compound token without decompounding. Recall degrades on compound queries but the pipeline runs. This is an LAWS-03 compliance degradation, not a blocker.

---

## Security Domain

Skipped — `security_enforcement` is not set in `.planning/config.json`; treating as not applicable for a competitive Kaggle ML pipeline that only reads public competition data, writes a local CSV, and ships no user-facing surface. The only secret involved is `kaggle.json` (API credentials) which Phase 1 does not modify.

---

## Sources

### Primary (HIGH confidence)

- BGE-M3 model card — `https://huggingface.co/BAAI/bge-m3` — 1024 dim, 8192 max length, fp16 support, no prefixes required
- BGE-M3 transformers loading (CLS pool, no prefix) — `https://huggingface.co/BAAI/bge-m3/discussions/17`
- BGE-M3 dense head clarification (last_hidden_state[:,0] NOT pooler_output) — `https://huggingface.co/BAAI/bge-m3/discussions/80`
- BGE-M3 prefix question thread — `https://huggingface.co/BAAI/bge-m3/discussions/35`
- bm25s project — `https://github.com/xhluca/bm25s` — accepts pre-tokenized list-of-list input, custom stemmer/stopword callable
- bm25s paper — `https://arxiv.org/abs/2407.03618` — 500× speedup over rank-bm25
- faiss-gpu-cu12 1.14.1.post1 — `https://pypi.org/project/faiss-gpu-cu12/` — Linux x86_64 only, CUDA 12.1+, Python 3.10-3.13, Compute Capability 7.0-8.9 (T4 is 7.5)
- Competition data spec — `./AboutData.md` — submission format, canonical citation strings are closed vocabulary, val/test distribution match
- CharSplit — `https://github.com/dtuggener/CharSplit` — German compound splitter, ~95% head-detection on Germanet
- MarianMT in transformers — `https://huggingface.co/docs/transformers/model_doc/marian` — AutoTokenizer + MarianMTModel loading pattern

### Secondary (MEDIUM confidence)

- `.planning/research/STACK.md` — BGE-M3 VRAM ~4-6GB fp16, bm25s speedup, faiss GPU rationale
- `.planning/research/PITFALLS.md` — AH-1 (silent CPU fallback), AH-3 (fixed count), AH-4 (e5-small), CP-1 (German decompounding), CP-4 (val overfitting), CP-6 (citation format), CP-8 (cross-encoder latency)
- `.planning/research/SUMMARY.md` — six-stage pipeline rationale, phase ordering justification
- GerDaLIR (German legal IR) — `https://github.com/lavis-nlp/GerDaLIR` — no-stemming guidance for German legal BM25
- bitsandbytes T4 compute capability — `https://github.com/TimDettmers/bitsandbytes/issues/529` — T4 = 7.5, fp16 only for 4-bit (relevant for Phase 5, not Phase 1)

### Tertiary (LOW confidence)

- `faiss-gpu-cu12` on Kaggle kernel specifically — no official confirmation, inferred from PyPI compatibility matrix + T4 CC
- CharSplit PyPI package name stability — checked but multiple variants exist (`CharSplit`, `charsplit`, `compound-split`); planner must pick one at install time
- Exact BGE-M3 encoding throughput on Kaggle T4 — extrapolated from `.planning/research/STACK.md`'s "500 docs/sec on T4" claim, not empirically measured
- OpusMT tc-big batch size 8 with beam 4 fits T4 — not benchmarked

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — BGE-M3, bm25s, faiss-gpu-cu12, mmarco, OpusMT-tc-big all have primary-source verification
- BGE-M3 pooling / prefix — HIGH — two separate HF discussions confirm CLS + no prefix
- Architecture patterns — HIGH — monolithic notebook matches existing `solution.py`; D-01 to D-07 are locked
- Runtime budget estimates — MEDIUM — based on published throughput claims; smoke test will verify on day 1
- Swiss citation canonicalization rules — MEDIUM — observed formats are verified from AboutData.md, but alias table completeness is unverified (A1-A4)
- CharSplit integration — LOW-MEDIUM — package exists, API documented, but exact PyPI name and Kaggle install path unverified

**Research date:** 2026-04-11
**Valid until:** 2026-05-11 (30 days; BGE-M3 + bm25s + faiss stack is stable; re-verify only if Kaggle kernel changes CUDA/torch versions)

---

*Researcher notes: This phase's risk is NOT about picking the right libraries — the stack is well-established and all four locked decisions (BGE-M3, bm25s, faiss-gpu-cu12, opus-mt-tc-big) match published best practice. The risk is about the SMALL correctness details — CLS vs mean pool, no prefixes, CharSplit API, canonicalizer idempotence, submission validator rules. The plan should dedicate a Wave-0-style "correctness smoke test" cell (the SMOKE block) that runs these assertions on tiny inputs before committing the full 15-minute encoding run.*
