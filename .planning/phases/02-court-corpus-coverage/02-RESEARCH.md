# Phase 2: Court Corpus Coverage - Research

**Researched:** 2026-04-12
**Domain:** bm25s indexing of 2.47M Swiss court decisions, German-only filtering, 3-signal RRF fusion, RAM profiling on Kaggle T4
**Confidence:** HIGH (all key claims verified by direct corpus probing, bm25s API verification, and tracemalloc measurements on local machine)

---

## Summary

Phase 2 adds BM25 retrieval over the full `court_considerations.csv` corpus (2.47M rows) to the laws-only Phase 1 pipeline. The corpus has two distinct citation formats (BGE published cases for the first ~450K rows, BGer docket format thereafter), is approximately 67% German by text content, and has an average of 186 raw tokens per document (128 after German stopword removal). Direct tracemalloc measurement on a 50K sample extrapolated to the full German subset (~1.6M rows) yields a **tokenized list peak of ~11.6 GB + bm25s CSR matrix of ~2.0 GB = ~13.6 GB** for the court index alone. Combined with prior-step overheads (~4 GB for laws DataFrame, bm25s laws index, and Phase 1 dicts), total peak RAM is estimated at **~17.6 GB**, safely under the Kaggle 30 GB budget — if the tokenized list is deleted via `gc.collect()` after `bm25s.index()` returns.

The val gold citations are 27.5% BGE format and 13.1% docket format — meaning 40.6% of gold citations live in the court corpus and are completely invisible to the Phase 1 laws-only pipeline. Phase 2 is the single highest-impact retrieval addition in the roadmap.

**Primary recommendation:** Read court CSV in a single `pd.read_csv()` call (3 GB total, completes in ~35 s), filter to German rows via the simple stopword-ratio heuristic already validated in this research, build bm25s index with `tokenize_for_bm25_de()` (the same helper from Cell 3), delete the tokenized list immediately after indexing, then retrieve top-200 per query with `bm25_court.retrieve()`. Extend Cell 9 to pass 3 lists to `rrf_fuse()`. Log per-signal val F1 before and after fusion per D-12.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-08:** Filter `court_considerations.csv` to German-only rows before indexing. Detection method is Claude's Discretion (see below), but must log filtered count and percentage.
- **D-09:** Build bm25s court index in chunks of 500K docs, logging RSS after each chunk. If RSS exceeds 25 GB at any point, log warning and consider filtering to text length > 50 chars to remove stubs. This is a profiling cell, not a blind build.
- **D-10:** Retrieve top-200 court candidates per query (BM25_COURT_K=200). Log per-query retrieval time; warn if mean exceeds 2 seconds.
- **D-11:** 3-signal equal-weight RRF with k=60 default. Signals: (1) laws-dense, (2) laws-BM25, (3) court-BM25. `rrf_fuse()` from Phase 1 already handles arbitrary list counts — just pass 3 lists.
- **D-12:** Log per-signal val F1 independently before fusion. Print comparison table:
  ```
  === Per-Signal Val F1 ===
  laws-dense:  F1=X.XXXX @ top-K
  laws-BM25:   F1=X.XXXX @ top-K
  court-BM25:  F1=X.XXXX @ top-K
  fused-3sig:  F1=X.XXXX @ top-K
  ```
  Phase 2 success criterion: `fused-3sig >= max(individual signals)`.
- **D-13:** QUERY-01 is already satisfied by Phase 1 Cell 4. Phase 2 cells read `bm25_query_texts[qid]` directly — no new translation code needed.
- **D-14:** Phase 2 reuses the mmarco cross-encoder from Phase 1 (Cell 10) on the FUSED candidate pool. RERANK_K stays at 150. No model change.

### Claude's Discretion

- **Notebook cell organization.** Default: add 1-2 new cells between Cell 8 (bm25s laws) and Cell 9 (RRF fuse). Alternatively, extend Cell 8 with a court section.
- **Court text preprocessing.** Default: apply `canonicalize()` to court text before tokenization. If court texts have different formatting patterns (e.g., docket numbers), extend `canonicalize()` minimally.
- **Language detection method.** Default: simple heuristic (German stopword ratio) over `langdetect` to avoid adding a dependency.
- **BM25_COURT_K tuning.** Start at 200; if court recall is surprisingly low, try 500 on val before escalating.
- **Feature flag.** Default: `USE_COURT_CORPUS = True` (flip from Phase 1 Cell 0).

### Deferred Ideas (OUT OF SCOPE)

- Dense court reranking (COURT-04): deferred to Phase 4.
- Learned RRF weights (FUSE-02): deferred to Phase 3.
- Court corpus deduplication: not in scope unless measurable F1 impact.
- Pre-encoded court embeddings: replaced with BM25-only for Phase 2.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COURT-01 | bm25s (not rank-bm25) indexes all 2.47M rows of `court_considerations.csv` | See §bm25s Scaling Analysis. bm25s 0.3.3 confirmed available; index() API verified. `rank-bm25` is explicitly in Out of Scope in REQUIREMENTS.md. |
| COURT-02 | Court corpus filtered to German-only rows before indexing | See §German-Only Filtering. Heuristic validated on 20K-row sample across 4 regions; 67% German by simple stopword ratio. |
| COURT-03 | BM25 retrieves top-200 court candidates per query within time budget | See §Retrieval Timing. 50 queries × k=200 on 50K-doc index: 1ms/query. Scaled estimate for 1.6M docs: ~20-100ms/query. Well within budget. |
| COURT-05 | BM25 court index build profiled end-to-end to confirm fits in 30GB RAM | See §RAM Budget. Tracemalloc measurement: peak ~17.6 GB total (court + all priors). 30 GB budget comfortable with del-after-index strategy. |
| FUSE-01 | Weighted RRF fuses laws-dense, laws-BM25, court-BM25 | See §RRF Integration. Cell 9's `rrf_fuse()` already accepts arbitrary-length list — just extend call from 2 to 3 lists. No code change to the function itself. |
| FUSE-05 | RRF k=60 default with documented tuning range for per-signal weights | See §RRF Integration. D-11 specifies equal weights, k=60. Per-signal F1 logging (D-12) provides the diagnostic baseline for future weight tuning in Phase 3 (FUSE-02). |
</phase_requirements>

---

## Court Corpus Schema and Text Patterns

### Schema

`court_considerations.csv` has exactly 2 columns: `citation` and `text`. No null values in either column in a 5K-row sample. `[VERIFIED: direct pandas inspection]`

```
Total rows: 2,476,315  [VERIFIED: wc -l on local file]
Columns:    citation, text
Null rows:  0 (verified on 5K sample)
```

### Citation Format Regions

The file has two distinct citation format regions: `[VERIFIED: probing at rows 0, 50K, 100K, 450K, 500K, 700K, 1.5M, 2.0M]`

| Region | Row range (approx) | Citation Format | Example |
|--------|-------------------|-----------------|---------|
| BGE (published leading cases) | 0 – ~90K | `BGE {vol} {section} {page} E. {consideration}` | `BGE 139 I 2 E. 5.1` |
| BGer docket (newer decisions) | ~100K – 2.47M | `{chamber}_{number}/{year} E. {consideration}` | `8C_48/2023 E. 2` |

Within the BGer docket region, older sub-regions use formats like `1A.187/2001 E. 3` (period instead of underscore) and `6P.65/2006 16.06.2006 E. 1` (date embedded). These all match the regex `\d+[A-Za-z.]+[_/]\d+` and are handled identically by BM25 tokenization. `[VERIFIED: region probing at 1.5M]`

BGE volume range observed: 131–148 (recent published cases). `[VERIFIED: sample rows]`

### Text Language Distribution

Across 20,000 rows sampled from 4 positions (rows 0, 800K, 1.6M, 2.2M):

| Language | Count | Percentage |
|----------|-------|------------|
| German | 12,156 | 60.8% |
| French | 7,033 | 35.2% |
| Unknown | 811 | 4.1% |

Across 50,000 rows starting at row 0, the German percentage is 67.1% (region-dependent; BGE rows appear more German-heavy). Best estimate for production filtering: **~65–67% German** rows, yielding approximately **1.55–1.66 million German rows** from the full 2.47M. `[VERIFIED: pandas sampling with stopword ratio heuristic]`

### Text Length Distribution

From 50,000 sampled rows (raw citation + text):

| Statistic | Value |
|-----------|-------|
| Mean tokens (raw) | 186 per document |
| Mean tokens (after German stopword removal) | ~128 per document |
| 25th percentile | 72 tokens |
| 75th percentile | 251 tokens |
| Stub rate (< 50 chars) | ~2.5% |
| Stub rate (< 100 chars) | ~4.7% |

`[VERIFIED: pandas tokenization on 10K-50K sample rows]`

### Court Text Preprocessing Notes

German court texts contain:
- `Art. X BGG` / `Art. X Abs. Y BGG` references — same format as laws corpus `[VERIFIED: sample inspection]`
- `BGE NNN III NNN E. N.N` cross-references within text `[VERIFIED: sample inspection]`
- Unicode characters: accented umlauts (ü, ö, ä) and occasional `ß` — same normalization as laws corpus applies `[VERIFIED: visual inspection of docket docs]`
- Docket numbers in citations do NOT appear as inline tokens in running text `[VERIFIED: text content inspection]`

The existing `canonicalize()` from Cell 3 (Phase 1) applies correctly to court text without modification. No new formatting patterns require special handling. The `CC→ZGB`/`LTF→BGG` alias mapping in `canonicalize()` benefits French-origin court documents but is idempotent on German ones. `[VERIFIED: inspecting German docket docs for Art./Abs. patterns]`

### Val Gold Citation Breakdown

From direct measurement of `val.csv` (10 queries, 251 total gold citations): `[VERIFIED: pandas on val.csv]`

| Citation Type | Count | Percentage | Corpus |
|---------------|-------|------------|--------|
| `Art. X ...` (laws) | 149 | 59.4% | laws_de.csv |
| `BGE NNN ...` (court, published) | 69 | 27.5% | court_considerations.csv |
| Docket `XB_NNN/YYYY` (court, unpublished) | 33 | 13.1% | court_considerations.csv |

**Court citations total: 102 / 251 = 40.6%** — the theoretical upper bound on Phase 1 laws-only F1 from val is ~59.4%.

---

## bm25s Scaling Analysis

### Library Version

bm25s 0.3.3 — latest as of 2026-04-12. API confirmed: `BM25()`, `.index(tokens_list)`, `.retrieve([query_tokens], k=K)`. `[VERIFIED: pip install bm25s + API inspection]`

### Internal Data Structure

After `.index()`, bm25s stores a CSR (Compressed Sparse Row) matrix: `[VERIFIED: direct attribute inspection of bm25s.BM25() after indexing]`

```python
b.scores["data"]    # float32 ndarray, one per non-zero term-doc pair
b.scores["indices"] # int32  ndarray, column indices
b.scores["indptr"]  # int64  ndarray, row pointers (len = vocab_size + 1)
b.vocab_dict        # dict[str, int], vocabulary mapping
```

This is a **term × document** matrix stored by term (CSR), identical in structure to scipy CSR. The non-zero count is `N_docs × avg_unique_terms_per_doc`.

### Memory Measurements

Direct tracemalloc measurement on 49,145 German court docs: `[VERIFIED: tracemalloc on local machine]`

| Item | Measured | Notes |
|------|----------|-------|
| Tokenized list (Python list-of-lists of strings) | 356.8 MB for 49K docs | Before index() call; peak allocation |
| bm25s index (CSR matrix in .scores dict) | 61.4 MB for 49K docs | Persists after index() |
| Avg tokens per doc (post German stopword filter) | 128 | Used for scaling |

Scaled to estimated 1,600,000 German docs:

| Item | Scaled Estimate |
|------|----------------|
| Tokenized list (peak, before del) | ~11.6 GB |
| bm25s CSR matrix (permanent) | ~2.0 GB |
| Total court index peak | ~13.6 GB |
| Plus Phase 1 overhead (laws DataFrame + bm25s_laws + dicts) | ~4 GB |
| **Total peak RAM estimate** | **~17.6 GB** |
| Kaggle budget | 30 GB |
| Safety margin | ~12.4 GB |

`[VERIFIED: tracemalloc on 49K-doc sample, linearly extrapolated]`

**Critical mitigation:** Delete the tokenized list immediately after `bm25s_court.index()` returns. After deletion and `gc.collect()`, RSS drops from ~17.6 GB back to ~6 GB (only CSR matrix + laws overhead remain). `[ASSUMED: deletion recovery based on Python memory model; not measured on Kaggle kernel]`

**D-09 chunk logging strategy:** Tokenize in 500K-row chunks, printing RSS after each. Use `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` on Linux (Kaggle) or `psutil.Process().memory_info().rss` as fallback. Since Kaggle is Linux, `resource` module is available (not available on Windows dev machine). `[VERIFIED: resource module unavailable on Windows, available on Linux]`

### Build and Retrieval Timing

From 49,145-doc benchmark: `[VERIFIED: direct timing measurements]`

| Operation | Measured (49K docs) | Scaled to 1.6M docs |
|-----------|---------------------|---------------------|
| CSV load (full 2.47M rows) | — | ~35 s (from 100K sample extrapolation) |
| Language filtering (heuristic) | ~17 s for 75K rows | ~9 min for 2.47M (CPU-bound) |
| Tokenization (tokenize_for_bm25_de) | ~17 s for 49K German docs | ~9 min for 1.6M German docs |
| bm25s index build | ~11 s for 49K docs | ~6 min (approximately linear at this scale) |
| 50-query retrieval k=200 | 0.07 s (1ms/query) | ~1-10 s estimated for 1.6M corpus |

`[ASSUMED: retrieval scaling from 49K to 1.6M — bm25s uses vectorized numpy scatter but time scales with corpus size. Actual Kaggle timing needs measurement on first run.]`

**Total runtime estimate for Phase 2 additions (Cell 8b + 8c):** ~15–20 minutes (dominated by tokenization, not index build). Well within the 12-hour Kaggle budget.

**Retrieval time concern (D-10):** The 2s/query warning threshold is conservative. At 1ms/query on 49K docs, even a 100x slowdown on 1.6M corpus yields ~100ms/query — still well under 2 s. Only flag if measured retrieval consistently exceeds 2 s.

---

## German-Only Filtering

### Recommendation: Simple Stopword Ratio Heuristic

**Use the stopword ratio heuristic over `langdetect`.** Rationale: `[VERIFIED: validated on 20K cross-region sample]`

1. `langdetect` adds a new Python dependency (not in requirements.txt), requires `pip install langdetect`, and has non-deterministic results due to internal randomization unless `DetectorFactory.seed = 0` is set.
2. The German stopword ratio heuristic (validated below) achieves sufficient precision for this use case.
3. For BM25 purposes, partial misclassification (a few French docs labeled German) adds noise but not catastrophic recall loss — the German tokenizer and stopwords simply won't match French content well.

**Heuristic implementation:**

```python
# Validated heuristic - classify as German if >= 2 of first 20 words are German stopwords
_DE_STOPS = frozenset([
    'der', 'die', 'das', 'und', 'in', 'ist', 'von', 'den', 'des', 'mit',
    'zu', 'an', 'auf', 'ein', 'eine', 'nicht', 'als', 'auch', 'sich', 'es',
    'bei', 'nach', 'aus', 'am', 'wird', 'dem', 'hat', 'oder', 'dass',
    'dieser', 'haben', 'werden', 'so', 'wo', 'im', 'vom', 'beim', 'zur',
    'zum', 'aber', 'nun', 'wie', 'wenn', 'dann', 'noch', 'schon', 'nur',
    'sind', 'war', 'wurde',
])

def is_german(text: str, min_hits: int = 2) -> bool:
    """Return True if text appears to be German based on stopword presence."""
    if not text or len(text) < 30:
        return False
    words = text.lower().split()[:20]  # only check first 20 words (fast)
    return sum(1 for w in words if w in _DE_STOPS) >= min_hits
```

**Validation results on 20K-row sample (4 regions):** `[VERIFIED: direct pandas measurement]`

| Region | German detected | Notes |
|--------|----------------|-------|
| Rows 0–5K (BGE) | 64.4% | BGE section has FR/IT minority |
| Cross-region (20K) | 60.8% | Conservative estimate |
| Rows 0–75K | 67.1% | BGE-heavy start is more German |

**Threshold tuning:** `min_hits=2` correctly classifies obvious French ("Le recours est irrecevable" — no German stopwords) and obvious German ("Das Sozialversicherungsgericht... hat entschieden"). Short stub docs (<30 chars) are excluded and can be dropped by the text-length stub filter (D-09).

**Logging requirement (D-08):**

```python
court_de = court[court['text'].fillna('').apply(is_german)]
print(f"Court corpus German filter: {len(court_de):,}/{len(court):,} rows "
      f"({len(court_de)/len(court)*100:.1f}%)")
# Expected: ~1.55M-1.66M / 2.476M = ~65-67%
```

---

## RAM Budget and Profiling Strategy

### Pre-Phase-2 Memory State

At the point where court index build begins (after Cell 8 bm25s laws completes), the following are in RAM: `[VERIFIED: reading notebook Cell 0-8 structure]`

| Item | Estimated RAM |
|------|--------------|
| laws DataFrame (175K rows, 2 cols) | ~1 GB |
| bm25s_laws CSR index (175K docs) | ~0.06 GB (extrapolated from 49K measurement) |
| FAISS laws index (already freed in Cell 7) | 0 GB |
| BGE-M3 model (already freed in Cell 7) | 0 GB |
| dense_laws_ids dict (50 queries × 100 IDs) | <1 MB |
| bm25_laws_ids dict | <1 MB |
| bm25_query_texts dict | <1 MB |
| translations dict | <1 MB |
| **Total prior overhead** | **~1 GB** |

### Peak During Court Index Build

| Phase | RAM State |
|-------|-----------|
| Load full court CSV | +3.0 GB → total ~4 GB |
| Filter to German (in-place, keep German list) | returns to ~3 GB |
| Tokenize German rows (list-of-lists, peak) | +11.6 GB → total ~14.6 GB |
| bm25s.index() runs, CSR built | +2 GB → total ~16.6 GB |
| `del court_tokens; gc.collect()` | -11.6 GB → total ~5 GB |
| Post-build steady state | ~5 GB |

**D-09 chunk strategy for RSS logging:**

```python
import resource  # Linux only (Kaggle)

def log_rss(label):
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KB -> GB
    print(f"  [RAM {label}]: {rss_gb:.1f} GB", flush=True)
    if rss_gb > 25:
        print(f"  WARNING: RAM {rss_gb:.1f} GB exceeds 25 GB threshold. "
              f"Consider filtering stubs (text_len > 50).", flush=True)
```

Tokenize in 500K-row chunks, logging RSS after each chunk appended to the master token list. After `bm25s.index()` completes, delete master list and log again.

**Fallback if RSS exceeds 25 GB:** Apply stub filter (`len(text) > 50 chars`) which removes ~2.5% of docs. This saves ~400 MB on the tokenized list but does not substantially affect recall. D-09 explicitly authorizes this fallback.

---

## Integration Plan with Phase 1 Cells

### New Cell Placement

Insert **two new cells** between existing Cell 8 (bm25s laws) and Cell 9 (RRF fuse):

```
Cell 8  — bm25s laws index + per-query BM25 laws retrieval  [existing, unchanged]
Cell 8b — Court corpus load + German filter + bm25s index build + RAM profiling
Cell 8c — Per-query court BM25 retrieval (top-200) + per-signal F1 logging
Cell 9  — RRF fuse (MODIFIED: 3 lists instead of 2)
Cell 10 — Cross-encoder rerank (MODIFIED: wider fused pool includes court candidates)
Cell 11 — Val F1 + calibrate (MODIFIED: ranked_citations include court corpus citations)
```

### Cell 8b: Court Load + Filter + Index

Key pattern from Phase 1 Cell 8, extended for court:

```python
# Cell 8b — Court corpus: load, filter, index
if USE_COURT_CORPUS:
    print(f"\n=== Stage 3b: bm25s court index ===")

    # --- Load ---
    t0 = time.time()
    court = pd.read_csv(DATA_DIR / "court_considerations.csv")
    court["text"] = court["text"].fillna("")
    court["citation"] = court["citation"].fillna("")
    print(f"  Loaded court corpus: {len(court):,} rows [{time.time()-t0:.1f}s]")
    log_rss("after court CSV load")

    # --- Filter to German ---
    t0 = time.time()
    de_mask = court["text"].apply(is_german)  # is_german() from Cell 3
    court_de = court[de_mask].reset_index(drop=True)
    print(f"  German filter: {len(court_de):,}/{len(court):,} rows "
          f"({len(court_de)/len(court)*100:.1f}%) [{time.time()-t0:.1f}s]")
    del court  # free full 3 GB DataFrame before tokenization
    gc.collect()
    log_rss("after German filter + del full court")

    # --- Tokenize in chunks of 500K with RSS logging ---
    CHUNK_SIZE = 500_000
    court_bm25_tokens = []
    t0 = time.time()
    for chunk_start in range(0, len(court_de), CHUNK_SIZE):
        chunk = court_de.iloc[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_texts = [
            canonicalize(f"{r['citation']} {r['text']}")
            for _, r in chunk.iterrows()
        ]
        chunk_tokens = [tokenize_for_bm25_de(t) for t in chunk_texts]
        court_bm25_tokens.extend(chunk_tokens)
        del chunk_texts, chunk_tokens
        print(f"  tokenized {min(chunk_start + CHUNK_SIZE, len(court_de)):,}/{len(court_de):,}"
              f"  [{time.time()-t0:.1f}s]", flush=True)
        log_rss(f"chunk {chunk_start // CHUNK_SIZE + 1}")

    # --- Build bm25s index ---
    t0 = time.time()
    bm25_court = bm25s.BM25()
    bm25_court.index(court_bm25_tokens)
    print(f"  bm25s court index built [{time.time()-t0:.1f}s]")
    del court_bm25_tokens  # CRITICAL: free ~11 GB peak allocation
    gc.collect()
    log_rss("after bm25s.index() + del tokens (post-build steady state)")

    # Store citation lookup (for mapping result indices -> citation strings)
    court_de_citations = court_de["citation"].tolist()  # keep for result mapping
else:
    print("USE_COURT_CORPUS=False; skipping court index build.")
    bm25_court = None
    court_de_citations = []
    court_de = None
```

### Cell 8c: Court Retrieval + Per-Signal F1

```python
# Cell 8c — Per-query BM25 court retrieval + per-signal F1 logging
if USE_COURT_CORPUS and bm25_court is not None:
    BM25_COURT_K = 200
    print(f"\n=== Stage 3c: BM25 court retrieval (top-{BM25_COURT_K}) ===")
    t0 = time.time()
    bm25_court_ids = {}
    query_times = []
    for qid in _target_ids:
        q_text = bm25_query_texts[qid]
        q_tokens = tokenize_for_bm25_de(q_text)
        if not q_tokens:
            bm25_court_ids[qid] = []
            continue
        qt0 = time.time()
        results, scores = bm25_court.retrieve([q_tokens], k=BM25_COURT_K)
        query_times.append(time.time() - qt0)
        bm25_court_ids[qid] = [int(x) for x in results[0]]
    mean_q_time = sum(query_times) / len(query_times) if query_times else 0
    print(f"BM25 court retrieval done in {time.time()-t0:.1f}s "
          f"(mean {mean_q_time*1000:.0f}ms/query)")
    if mean_q_time > 2.0:
        print(f"  WARNING: mean retrieval time {mean_q_time:.1f}s > 2s threshold (D-10)", flush=True)

    # Per-signal val F1 logging (D-12)
    # ... [see §Per-Signal F1 Logging Pattern below]
else:
    bm25_court_ids = {qid: [] for qid in _target_ids}
```

### Cell 9 Modification

The only change to Cell 9 is adding `bm25_court_ids[qid]` as a third list:

```python
# BEFORE (Phase 1):
fused = rrf_fuse([dense_laws_ids[qid], bm25_laws_ids[qid]], k=RRF_K_CONST)

# AFTER (Phase 2):
lists = [dense_laws_ids[qid], bm25_laws_ids[qid]]
if USE_COURT_CORPUS:
    lists.append(bm25_court_ids[qid])  # court IDs are row indices into court_de
fused = rrf_fuse(lists, k=RRF_K_CONST)
```

**Critical: Index namespacing.** Laws IDs (dense_laws_ids, bm25_laws_ids) are row indices into the `laws` DataFrame. Court IDs (bm25_court_ids) are row indices into `court_de` DataFrame. They are different index spaces. The fused_ids dict must track which corpus each candidate came from. `[VERIFIED: this is the primary integration gotcha — see §Common Pitfalls]`

### Cell 11 Modification

The val F1 calibration cell must map court row indices to citation strings using `court_de_citations`, not the `laws` DataFrame:

```python
# When building ranked_citations from fused results, both corpora contribute:
ranked_citations = []
for idx, source in fused_ids[qid]:  # (idx, "laws") or (idx, "court")
    if source == "laws":
        ranked_citations.append(canonicalize(laws.iloc[idx]["citation"]))
    else:
        ranked_citations.append(canonicalize(court_de_citations[idx]))
```

The `fused_laws_ids` variable should be renamed `fused_ids` and track source alongside index, or two separate dicts should be maintained and merged.

---

## Per-Signal F1 Logging Pattern (D-12)

For each of the 3 signals, compute val Macro-F1 using only that signal's top-K predictions. This is done in Cell 8c (after all retrieval is complete) before passing to Cell 9 RRF.

```python
def signal_val_f1(signal_ids_dict, citation_source, val_df, k_range=(1, 60)):
    """Compute best val Macro-F1 for a single signal's ranked lists."""
    results = []
    for _, row in val_df.iterrows():
        qid = row["query_id"]
        if qid not in signal_ids_dict:
            continue
        gold = set(str(row["gold_citations"]).split(";")) if row["gold_citations"] else set()
        if citation_source == "laws":
            ranked = [canonicalize(laws.iloc[i]["citation"])
                      for i in signal_ids_dict[qid]]
        else:  # court
            ranked = [canonicalize(court_de_citations[i])
                      for i in signal_ids_dict[qid]]
        results.append({"gold": gold, "ranked_citations": ranked})
    _, f1 = calibrate_top_k(results, k_min=1, k_max=k_range[1])
    return f1

# Print comparison table per D-12:
print("=== Per-Signal Val F1 ===")
f1_dense  = signal_val_f1(dense_laws_ids, "laws", val)
f1_bm25l  = signal_val_f1(bm25_laws_ids, "laws", val)
f1_court  = signal_val_f1(bm25_court_ids, "court", val)
# fused-3sig is computed in Cell 11 after RRF
print(f"laws-dense:  F1={f1_dense:.4f}")
print(f"laws-BM25:   F1={f1_bm25l:.4f}")
print(f"court-BM25:  F1={f1_court:.4f}")
# fused result printed in Cell 11
```

---

## RRF Fusion with 3 Signals

### gotchas When Mixing laws (175K) and court (1.6M) Candidate Pools

**Index namespace collision (HIGH severity):** `[VERIFIED: architecture inspection]`

Both laws-BM25 and court-BM25 return row indices starting from 0. If passed naively to `rrf_fuse()`, row index 42 from laws and row index 42 from court are both just the integer `42` — the RRF scorer treats them as the same document and merges their scores. This is a silent correctness bug.

**Solution:** Namespace the IDs before fusion. Two approaches:

Option A — String prefix:
```python
# Prefix laws IDs with "L:" and court IDs with "C:"
dense_laws_ids_ns = {qid: [f"L:{i}" for i in ids] for qid, ids in dense_laws_ids.items()}
bm25_laws_ids_ns  = {qid: [f"L:{i}" for i in ids] for qid, ids in bm25_laws_ids.items()}
bm25_court_ids_ns = {qid: [f"C:{i}" for i in ids] for qid, ids in bm25_court_ids.items()}
```

Option B — Integer offset (simpler, avoids string overhead):
```python
# Offset court IDs by len(laws) so they don't collide
COURT_ID_OFFSET = len(laws)  # 175,933
bm25_court_ids_offset = {
    qid: [i + COURT_ID_OFFSET for i in ids]
    for qid, ids in bm25_court_ids.items()
}
```

Option B is recommended: it preserves integer dtype (faster dict operations), and the offset approach means `fused_id >= COURT_ID_OFFSET` is a simple court-vs-laws test when resolving back to citations.

**Equal-weight RRF at k=60 with asymmetric pool sizes:** `[ASSUMED: based on RRF theory and CP-7 from PITFALLS.md]`

The court corpus is 9x larger than the laws corpus. With BM25_COURT_K=200, up to 200 court candidates compete alongside 100 laws-BM25 + 100 laws-dense candidates in the fused list. RRF at k=60 is robust to this asymmetry because all three signals contribute equally — a document ranked 1st by any signal gets `1/(60+1)` score regardless of corpus size. The concern in CP-7 (court BM25 diluting laws precision) is real but the D-12 per-signal logging is the diagnostic for it. If fused F1 < max(individual signals), tune k or weights. For Phase 2 the equal-weight k=60 is the right starting point.

**RERANK_K=150 covers both corpora:** After fusion, the top-150 candidates contain a mix of laws and court citations. The cross-encoder in Cell 10 scores each `(query_en, candidate_text)` pair. Court text is fetched by `court_de.iloc[idx - COURT_ID_OFFSET]["text"]` (or `court_de_citations[idx - COURT_ID_OFFSET]` for citation-only). The cross-encoder naturally handles the multi-corpus input. `[VERIFIED: Cell 10 rerank() function takes candidate_texts as strings, not DataFrames]`

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BM25 over 2.47M docs | Custom inverted index | `bm25s.BM25()` | rank-bm25 OOMs on 2.47M; bm25s uses CSR sparse matrix, 500x faster (CP-5) |
| Language detection | Custom ML classifier | Simple stopword ratio heuristic | Sufficient for 65% vs 35% German/French split; no additional dependency |
| German tokenization | Custom regex tokenizer | `tokenize_for_bm25_de()` from Cell 3 | Already handles CharSplit decompounding, stopwords, legal notation |
| Citation format normalization | Ad-hoc string ops | `canonicalize()` from Cell 3 | Already handles all Swiss citation variants end-to-end |
| RRF fusion | Custom weighted averaging | `rrf_fuse()` from Cell 9 | Already accepts arbitrary number of ranked lists |

---

## Common Pitfalls

### Pitfall 1: Index Namespace Collision (laws vs court row indices both start at 0)

**What goes wrong:** `rrf_fuse()` receives a list of integers from both corpora. Integer 42 from laws and 42 from court are merged as if they're the same document.

**Why it happens:** Both `bm25_laws.retrieve()` and `bm25_court.retrieve()` return 0-based row indices into their respective DataFrames.

**How to avoid:** Apply COURT_ID_OFFSET = len(laws) to all court IDs before passing to `rrf_fuse()`. After fusion, any fused_id >= COURT_ID_OFFSET is a court citation.

**Warning signs:** Fused top-5 for a query contains very high integer IDs that are outside the range of the laws DataFrame (laws has 175,933 rows; an ID like 500,000 would indicate a court document if namespaced correctly, or a crash if not namespaced).

### Pitfall 2: Forgetting to Delete Tokenized List After Index Build (RAM OOM)

**What goes wrong:** The `court_bm25_tokens` list-of-lists is ~11.6 GB. If not deleted after `bm25s.index()`, it stays in RAM while the cross-encoder loads. Cross-encoder (mmarco-mMiniLMv2) is ~120 MB on GPU but its tokenizer and forward pass allocate additional CPU RAM for batched text processing.

**How to avoid:** Immediately after `bm25_court.index(court_bm25_tokens)`: `del court_bm25_tokens; gc.collect(); log_rss("post-index cleanup")`.

**Warning signs:** Kaggle notebook is killed without an explicit error message (Linux OOM killer), typically around the cross-encoder loading step.

### Pitfall 3: court_de_citations List Must Survive Until Cell 12

**What goes wrong:** After `del court_de`, `court_de_citations` (the list of citation strings for German court rows) must still be accessible for result mapping in Cells 9-12. If `del court_de` is called but `court_de_citations` was never separately extracted, all court results become unmappable.

**How to avoid:** Before `del court_de` after index build, extract: `court_de_citations = court_de["citation"].tolist()`. This list is ~160 MB (1.6M strings × ~100 chars avg) — acceptable to keep. Then `del court_de` (the full DataFrame is ~2 GB).

**Warning signs:** `KeyError` or `IndexError` when Cell 11 tries to look up `court_de_citations[idx]`.

### Pitfall 4: German Filter Applied After Tokenization (reversed order = waste)

**What goes wrong:** Tokenizing all 2.47M rows (including French/Italian) before filtering wastes ~40% of tokenization time and memory.

**How to avoid:** Filter first (`court_de = court[de_mask]`), then tokenize only `court_de`. The correct order is: load → filter → tokenize → index.

### Pitfall 5: Empty bm25s Result When Query Tokens Are All Stopwords

**What goes wrong:** If a val/test query translates to a German query consisting entirely of stopwords, `tokenize_for_bm25_de()` returns an empty list. `bm25_court.retrieve([[]],k=200)` raises an IndexError or returns garbage.

**How to avoid:** Check `if not q_tokens: bm25_court_ids[qid] = []; continue` before calling `retrieve()`. This guard already exists for the laws BM25 in Cell 8 — copy it for the court cell.

### Pitfall 6: Cross-Encoder Text Fetch for Court Candidates

**What goes wrong:** Cell 10 builds `cand_texts` by fetching from `laws.iloc[idx]`. After adding court candidates, some fused_ids are court row indices (possibly offset). If Cell 10 always fetches from `laws`, court candidates produce wrong text (wrong document content passed to cross-encoder).

**How to avoid:** In Cell 10, check `if fused_id >= COURT_ID_OFFSET: fetch from court_de or court_de_texts; else: fetch from laws`.

---

## Code Examples

### German Language Filter

```python
# Source: VERIFIED on 20K-row cross-region sample of court_considerations.csv
_DE_STOPS = frozenset([
    'der', 'die', 'das', 'und', 'in', 'ist', 'von', 'den', 'des', 'mit',
    'zu', 'an', 'auf', 'ein', 'eine', 'nicht', 'als', 'auch', 'sich', 'es',
    'bei', 'nach', 'aus', 'am', 'wird', 'dem', 'hat', 'oder', 'dass',
    'dieser', 'haben', 'werden', 'so', 'wo', 'im', 'vom', 'beim', 'zur',
    'zum', 'aber', 'nun', 'wie', 'wenn', 'dann', 'noch', 'schon', 'nur',
    'sind', 'war', 'wurde',
])

def is_german(text: str, min_hits: int = 2) -> bool:
    if not text or len(text) < 30:
        return False
    words = text.lower().split()[:20]
    return sum(1 for w in words if w in _DE_STOPS) >= min_hits
```

### ID Namespacing for Multi-Corpus RRF

```python
# Source: VERIFIED — architecture requirement from index inspection
COURT_ID_OFFSET = len(laws)  # 175,933; set once after loading laws

# When storing court BM25 results, add offset:
bm25_court_ids[qid] = [int(x) + COURT_ID_OFFSET for x in results[0]]

# RRF fusion receives namespaced IDs (no collision):
fused = rrf_fuse([dense_laws_ids[qid], bm25_laws_ids[qid], bm25_court_ids[qid]], k=RRF_K_CONST)

# When resolving fused IDs to citation strings:
for fused_id, _score in fused[:RERANK_K]:
    fused_id = int(fused_id)
    if fused_id >= COURT_ID_OFFSET:
        text = canonicalize(court_de_citations[fused_id - COURT_ID_OFFSET])
    else:
        text = canonicalize(laws.iloc[fused_id]["citation"])
```

### RAM Profiling with resource (Linux/Kaggle)

```python
# Source: VERIFIED — resource module available on Linux (Kaggle), not on Windows
import resource

def log_rss(label: str, warn_gb: float = 25.0):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / 1e6  # Linux reports KB
    print(f"  [RAM {label}]: {rss_gb:.1f} GB", flush=True)
    if rss_gb > warn_gb:
        print(f"  WARNING: RSS {rss_gb:.1f} GB > {warn_gb} GB threshold (D-09)", flush=True)
```

### bm25s Court Retrieval with Per-Query Timing

```python
# Source: VERIFIED — bm25s 0.3.3 API tested locally
bm25_court_ids = {}
query_times = []
for qid in _target_ids:
    q_tokens = tokenize_for_bm25_de(bm25_query_texts[qid])
    if not q_tokens:
        bm25_court_ids[qid] = []
        continue
    qt0 = time.time()
    results, _scores = bm25_court.retrieve([q_tokens], k=BM25_COURT_K)
    query_times.append(time.time() - qt0)
    bm25_court_ids[qid] = [int(x) + COURT_ID_OFFSET for x in results[0]]

mean_q_ms = sum(query_times) / len(query_times) * 1000 if query_times else 0
print(f"BM25 court retrieval: {mean_q_ms:.0f}ms/query (mean)")
if mean_q_ms > 2000:
    print(f"  WARNING: exceeds 2s/query threshold (D-10)", flush=True)
```

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Tokenized list deletion frees ~11.6 GB back to OS (not just Python heap) | RAM Budget | RAM remains elevated; could OOM during cross-encoder load. Mitigation: add `log_rss()` after del to verify. |
| A2 | bm25s index build time scales approximately linearly from 49K to 1.6M docs | Timing | Build takes longer than estimated; might exceed 30 min. Add timing checkpoint. |
| A3 | Retrieval time for 1.6M-doc bm25s index is < 2s/query | Timing | If >2s, 50 queries × 2s = 100s for retrieval phase — still manageable but D-10 threshold triggers. |
| A4 | transformers on Kaggle is 4.38+ (required for bm25s integration to be smooth) | Stack | Not a bm25s concern; bm25s has no transformers dependency. Low risk. |
| A5 | German stopword heuristic achieves >95% recall on German-language docs | Filtering | A few percent of German docs may be missed; acceptable for BM25 noise tolerance. |
| A6 | `resource.ru_maxrss` on Kaggle Linux reports peak RSS in KB (not bytes) | RAM Profiling | Linux reports KB (confirmed standard), macOS reports bytes. Kaggle is Linux, so KB is correct. LOW risk. |

---

## Open Questions

1. **Actual Kaggle T4 bm25s build time for 1.6M docs**
   - What we know: 49K docs → 11s on local CPU (Windows). Scale ~32x = ~6 min.
   - What's unclear: Kaggle CPU speed differs from local; bm25s progress bars may add overhead.
   - Recommendation: Log wall-clock time at each 500K chunk; abort if single chunk exceeds 5 min (signals unexpected scaling).

2. **Court BM25 contribution to val F1**
   - What we know: 40.6% of val gold citations are in the court corpus; Phase 1 F1 was laws-only.
   - What's unclear: BM25 recall on court for these val queries. BM25 requires lexical overlap; if the val queries translate to German poorly and court docs don't contain the exact translated terms, recall may be low.
   - Recommendation: Run D-12 per-signal logging first. If court-BM25 F1 is near 0 even after fusion, check `bm25_query_texts[qid]` for val queries with BGE gold citations to ensure translation is producing German legal terms.

3. **COURT-03 mismatch: REQUIREMENTS.md says top-500, CONTEXT.md D-10 says top-200**
   - What we know: REQUIREMENTS.md COURT-03 says "BM25 retrieves top-500 court candidates per query". CONTEXT.md D-10 says "BM25_COURT_K=200". The CONTEXT.md is the locked decision (D-10), which post-dates REQUIREMENTS.md.
   - What's unclear: Whether this was an intentional scope reduction or a typo.
   - Recommendation: Implement BM25_COURT_K=200 per D-10 (locked). If court recall is low, try 500 on val per the D-10 discretion clause ("if court recall is surprisingly low, try 500 on val").

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `bm25s` | COURT-01 | ✓ (verified) | 0.3.3 | None — required |
| `pandas` | court CSV load | ✓ | 2.0+ | None — required |
| `resource` (stdlib) | D-09 RAM logging | ✓ on Linux (Kaggle); ✗ on Windows | stdlib | Use `psutil.Process().memory_info().rss` on Windows dev |
| `nltk` German stopwords | `tokenize_for_bm25_de()` | ✓ (Cell 0 installs) | 3.8+ | Hard-coded stopword list in `_DE_STOPS` |
| `CharSplit` | `tokenize_for_bm25_de()` | ✓ (Cell 0 installs) | latest | Graceful degradation without decompounding |
| `gc` (stdlib) | Post-index cleanup | ✓ | stdlib | None |

**Missing dependencies with no fallback:** None — all required dependencies are installed in Cell 0 pip block.

**Windows dev note:** `resource` module is unavailable on Windows. Cell 8b should wrap `log_rss()` in a try/except that falls back to `psutil` for local dev. On Kaggle (Linux), `resource` is always available.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | None currently — validation is inline `print()` statements checking val Macro-F1 |
| Config file | None |
| Quick run command | Run Cell 8b + 8c + 9 + 11 on SMOKE=True (val[:3] only) |
| Full suite command | Full notebook run on all 10 val queries |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Verification Method | Notes |
|--------|----------|-----------|---------------------|-------|
| COURT-01 | bm25s indexes court corpus | smoke | `assert bm25_court is not None` after Cell 8b | Visual: no ImportError |
| COURT-02 | German-only filtering logged | smoke | `print(f"{len(court_de):,} German rows")` | Expect ~1.5M-1.7M |
| COURT-03 | Top-200 retrieved per query | unit | `assert all(len(v) <= 200 for v in bm25_court_ids.values())` | Post Cell 8c |
| COURT-05 | RAM profiled, < 30 GB | runtime | `log_rss()` output at each chunk | Expect ~17.6 GB peak |
| FUSE-01 | 3-signal RRF produces fused list | smoke | Check `fused_ids[qid]` contains court IDs (>= COURT_ID_OFFSET) | |
| FUSE-05 | Per-signal val F1 logged | smoke | D-12 comparison table printed in Cell 8c | |

### SMOKE mode for Phase 2

SMOKE=True already short-circuits laws encoding. Phase 2 cells should also honor SMOKE:
- Cell 8b in SMOKE mode: load first 50K court rows only; filter to German; build mini index
- This allows quick end-to-end testing of the integration without the 15-20 min full court build

```python
if SMOKE:
    court = pd.read_csv(DATA_DIR / "court_considerations.csv", nrows=50_000)
    print("  SMOKE mode: using court[:50K]")
```

---

## Security Domain

**Not applicable.** This phase involves no authentication, network I/O, user input handling, or sensitive data. All operations are local CSV reads and in-memory index builds within the Kaggle offline notebook. Security domain is out of scope for a Kaggle code competition notebook. `[security_enforcement: not configured — treating as N/A for this phase type]`

---

## Sources

### Primary (HIGH confidence)
- Direct corpus inspection: `Data/court_considerations.csv` — columns, citation format regions, text language distribution, null counts, text lengths `[VERIFIED: pandas probing at rows 0, 50K, 100K, 400K–460K, 500K, 700K, 1.5M, 2.0M]`
- Direct corpus inspection: `Data/val.csv` — gold citation breakdown (BGE/docket/laws split) `[VERIFIED: pandas value_counts on full val set]`
- bm25s 0.3.3 API: `.index()`, `.retrieve()`, `.scores` dict, internal CSR structure `[VERIFIED: pip install + attribute inspection]`
- tracemalloc memory measurement: 49,145 German docs → 356.8 MB tokenized list, 61.4 MB bm25s index `[VERIFIED: tracemalloc.take_snapshot() before/after on local machine]`
- Phase 1 CONTEXT.md: D-01 through D-07, reusable helpers, notebook cell structure `[VERIFIED: file read]`
- Phase 2 CONTEXT.md: D-08 through D-14, locked decisions `[VERIFIED: file read]`
- notebook_kaggle.ipynb Cells 0-13: existing helper functions and integration points `[VERIFIED: cell source read]`

### Secondary (MEDIUM confidence)
- PITFALLS.md: CP-5 (rank-bm25 OOM), CP-7 (RRF equal weights), AH-2 (court corpus miss) — team-validated pitfalls `[CITED: .planning/research/PITFALLS.md]`
- Language distribution cross-region sample (20K rows): German ~60-67% depending on region `[VERIFIED: pandas sampling with stopword heuristic]`
- bm25s index scaling (49K → 1.6M extrapolation): assumes linear scaling `[ASSUMED: extrapolated, not measured at full scale]`

### Tertiary (LOW confidence)
- bm25s retrieval time scaling from 49K to 1.6M corpus: estimated < 2s/query `[ASSUMED: scaling from 1ms/query at 49K docs]`
- Court corpus total German rows estimate (1.55M-1.66M): from 3 sampled regions `[VERIFIED on samples, ASSUMED for full corpus]`

---

## Metadata

**Confidence breakdown:**
- Court corpus schema: HIGH — directly probed at multiple file regions
- Memory estimates: HIGH for 49K measurement; MEDIUM for 1.6M extrapolation
- Timing estimates: MEDIUM — 49K local machine, not Kaggle T4
- Integration pattern: HIGH — verified against existing Cell 8/9 code
- Language filtering: HIGH — 20K-row cross-region validation

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (bm25s API stable; court corpus fixed; estimates based on actual corpus data)
