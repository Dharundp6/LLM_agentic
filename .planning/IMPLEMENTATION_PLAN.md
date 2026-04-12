# Implementation Plan: Pipeline Rebuild

**Date:** 2026-04-12
**Current Score:** 0.00952 (public LB) | **Target:** 0.30+ (top 5) | **Leaderboard #1:** 0.359
**Reference Paper:** Hybrid GraphRAG (0.691 val F1, 111% over BM25 baseline)

---

## Root Cause Analysis

Our F1=0.01 vs reference BM25 baseline=0.327 — a **33x gap on fundamentals**.

| Root Cause | Evidence | Impact |
|-----------|----------|--------|
| **OpusMT translation is catastrophic** | CLIRudit paper: OpusMT BLEU=10.77 vs GPT-4o=34.41. Our BM25 queries are garbage German. | **PRIMARY** — kills BM25 entirely |
| **No entity direct lookup** | Query says "Art. 221 StPO" but we search instead of just looking it up | HIGH — free precision |
| **No German stemming in BM25** | Reference uses PyStemmer; we tokenize without stemming | MEDIUM — misses morphological variants |
| **Court BM25 = pure noise** | Per-signal F1: court-BM25=0.0000, laws-dense=0.0246 | MEDIUM — dilutes best signal |
| **mmarco cross-encoder too weak** | General-purpose, not legal-domain; BGE-reranker-v2-m3 gives +0.069 F1 in ablation | MEDIUM |
| **Fixed-K calibration on 10 queries** | Massive overfitting risk; no elbow detection or LLM-estimated count | LOW-MEDIUM |

---

## Reference Paper Ablation (our target trajectory)

| Stage | F1 | Delta | What |
|-------|-----|-------|------|
| BM25 only | 0.327 | — | German stemming + good translations |
| + Semantic (BGE-M3 RRF) | 0.489 | +0.162 | Dense retrieval fused with BM25 |
| + Graph (PPR + communities) | 0.543 | +0.054 | Citation graph, co-citation, PPR |
| + Cross-encoder (BGE-reranker-v2) | 0.612 | +0.069 | Better reranker |
| + LLM verify | 0.658 | +0.046 | Qwen2.5-7B scores candidates |
| + Adaptive citation count | 0.691 | +0.033 | Elbow + LLM + calibrated blend |

---

## Rebuild Stages (Priority Order)

### Stage 1: Fix BM25 Foundation (0.01 → 0.15-0.25)

**Problem:** OpusMT produces garbage German. BM25 can't match anything.

**Fix 1a: Replace OpusMT with LLM translation**
- Use Qwen2.5-7B (already in our Phase 5 plan) for EN→DE query translation
- CLIRudit shows LLM translation (Llama 3.2B, BLEU=31.27) is 3x better than OpusMT (BLEU=10.77)
- Prompt: `"Translate the following English legal question to German:\n{query}\nGerman:"`
- Load Qwen2.5-7B 4-bit, translate all 50 queries (~2 min), unload, proceed
- **Alternative if VRAM tight:** Use BGE-M3 dense WITHOUT translation (cross-lingual) + BM25 with LLM-translated queries only

**Fix 1b: Add German stemming to BM25**
```python
import Stemmer
stemmer = Stemmer.Stemmer('german')
def tokenize_bm25_stemmed(text):
    tokens = tokenize_for_bm25_de(text)  # existing tokenizer
    return [stemmer.stemWord(t) if len(t) > 3 and '.' not in t else t for t in tokens]
```
- Install PyStemmer (pip install PyStemmer, available on Kaggle)
- Apply to BOTH corpus tokenization AND query tokenization
- Reference paper uses this — their BM25 baseline alone = 0.327

**Fix 1c: Entity direct lookup (bypass retrieval)**
```python
# For each query, extract explicit citations and force-include them
for qid in target_ids:
    query = q_en_canon[qid]
    # Match "Art. X [Abs. Y] CODE" patterns against laws['citation']
    for match in re.finditer(r'Art\.\s*\d+(?:\s+Abs\.\s*\d+)?(?:\s+lit\.\s*[a-z])?\s+[A-Z][A-Za-z]{1,6}', query):
        citation_str = canonicalize(match.group(0))
        exact = laws[laws['citation'].str.contains(citation_str, na=False)]
        if len(exact) > 0:
            entity_lookup_ids[qid].extend(exact.index.tolist())
```
- Weight 3.0 in RRF (per reference paper)
- This alone could add 5-12 correct citations per query

**Fix 1d: Multiple BM25 query variants**
- Reference paper runs 5+ BM25 queries per question:
  1. German translation (primary)
  2. Alternative MT translation
  3. LLM-generated query variants (up to 3)
  4. Extracted legal concepts
  5. HyDE-generated text
- Each returns top-100, all fused via weighted RRF

**Expected impact:** 0.01 → 0.15-0.25

---

### Stage 2: Upgrade Dense Retrieval (0.25 → 0.35)

**Fix 2a: BGE-M3 dense + sparse from same model**
```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Encode corpus with dense + sparse
output = model.encode(corpus_texts, return_dense=True, return_sparse=True)
dense_vecs = output['dense_vecs']      # FAISS IndexFlatIP
sparse_weights = output['lexical_weights']  # learned term weights

# Score combination: s = 1.0*dense + 0.3*sparse (from paper Table)
```
- Sparse signal is weak cross-lingually (43.3 vs 76.2 for dense on MKQA-de)
- BUT: for German-translated queries against German corpus, sparse adds value
- Use sparse on TRANSLATED queries only (same-language matching)

**Fix 2b: HyDE query expansion**
```python
# Generate hypothetical German legal passage
hyde_prompt = (
    "Please write a passage in German from a Swiss court decision "
    "or law article to answer the question.\n"
    f"Question: {query_en}\nPassage:"
)
hyde_text = qwen_generate(hyde_prompt, max_tokens=256)
hyde_embedding = bge_m3.encode([hyde_text], return_dense=True)['dense_vecs'][0]

# Combine with original query embedding
final_query_vec = (query_embedding + hyde_embedding) / 2  # Equation 8
```
- HyDE gives +38% nDCG on web search, +37-57% on multilingual
- Generates the doc IN GERMAN, matching corpus language
- Encoder's "lossy compression" filters LLM hallucinations

**Fix 2c: Dual-query dense retrieval**
- Search with BOTH English query embedding AND German translation embedding
- Average the two result sets or RRF-fuse them
- Already partially implemented (Cell 7 does dual-query)

**Expected impact:** 0.25 → 0.35

---

### Stage 3: Citation Graph + PPR (0.35 → 0.45)

**Fix 3a: Build citation graph from court corpus (offline, regex-only)**
```python
# Scan 2.47M court texts for statute references
import re
CITE_PATTERN = r'Art\.\s*\d+(?:\s+Abs\.\s*\d+)?\s+(?:ZGB|OR|StGB|StPO|BGG|SchKG|BV|DBG|MWSTG|ZPO|AHV|IVG|KVG|UVG|BVG)'
BGE_PATTERN = r'BGE\s+\d+\s+[IVX]+[a-z]?\s+\d+'

statute_to_courts = defaultdict(list)  # "Art. 221 StPO" → [court_doc_ids]
co_citation = defaultdict(int)  # ("Art. 221 StPO", "Art. 222 StPO") → count

for idx, row in court_df.iterrows():
    text = row['text']
    cited_statutes = set(re.findall(CITE_PATTERN, text))
    cited_statutes |= set(re.findall(BGE_PATTERN, text))
    for statute in cited_statutes:
        statute_to_courts[canonicalize(statute)].append(idx)
    # Co-citation pairs
    statutes_list = list(cited_statutes)
    for i in range(len(statutes_list)):
        for j in range(i+1, len(statutes_list)):
            pair = tuple(sorted([canonicalize(statutes_list[i]), canonicalize(statutes_list[j])]))
            co_citation[pair] += 1
```
- **No LLM needed** — regex extraction is sufficient for Swiss legal citations
- Pre-compute offline, upload as Kaggle dataset (pickle/json)
- ~10 min to scan 2.47M docs

**Fix 3b: Personalized PageRank**
```python
import scipy.sparse as sp
import numpy as np

# Build sparse adjacency matrix: statutes + court decisions as nodes
# Edges: court_decision → statute (citation edges)
N = len(laws) + len(court_df)
rows, cols = [], []
for statute, court_ids in statute_to_courts.items():
    statute_idx = laws[laws['citation'] == statute].index
    if len(statute_idx) > 0:
        for cid in court_ids:
            rows.append(cid + len(laws))  # offset court IDs
            cols.append(statute_idx[0])

A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
A = A + A.T  # undirected
# Normalize columns
col_sums = np.array(A.sum(axis=0)).flatten()
col_sums[col_sums == 0] = 1
A = A @ sp.diags(1.0 / col_sums)

# PPR: v_{t+1} = alpha * A * v_t + (1-alpha) * personalization
alpha = 0.5
def ppr(seed_indices, alpha=0.5, max_iter=50):
    p = np.zeros(N)
    for idx in seed_indices:
        p[idx] = 1.0 / len(seed_indices)
    v = p.copy()
    for _ in range(max_iter):
        v = alpha * A.dot(v) + (1 - alpha) * p
    return v
```
- Use scipy sparse, NOT NetworkX (2.47M nodes)
- Seed nodes = initial retrieval results (top-20 from dense+BM25)
- PPR output → rank documents by score → additional RRF signal

**Fix 3c: Co-citation expansion**
```python
# When statute A is retrieved, find frequently co-cited statutes
def expand_co_citations(statute_citation, top_n=10):
    related = []
    for (a, b), count in co_citation.items():
        if a == statute_citation:
            related.append((b, count))
        elif b == statute_citation:
            related.append((a, count))
    return sorted(related, key=lambda x: x[1], reverse=True)[:top_n]
```

**Fix 3d: Cross-signal boosting (w=5.0)**
```python
# Documents appearing in BOTH content-based AND graph-based signals
content_docs = set(dense_results + bm25_results)
graph_docs = set(ppr_results + co_citation_results)
cross_signal = content_docs & graph_docs
# Give these docs weight 5.0 in RRF
```
- Reference paper's HIGHEST weight signal
- "Documents confirmed by both content and graph retrieval are almost always relevant"

**Expected impact:** 0.35 → 0.45

---

### Stage 4: Better Reranking (0.45 → 0.55)

**Fix 4a: Replace mmarco with BGE-reranker-v2-m3**
- Reference paper uses BGE-reranker-v2-m3 (568M params)
- Cross-encoder gives +0.069 F1 in their ablation
- Model: `BAAI/bge-reranker-v2-m3` on HuggingFace
- OR: Use BGE-M3 ColBERT mode as reranker (free if model already loaded):
```python
scores = model.compute_score(
    [[query, doc] for doc in top_300_candidates],
    weights_for_different_modes=[0.4, 0.2, 0.4]  # dense, sparse, colbert
)
```

**Fix 4b: Rerank top-300 (not top-150)**
- Reference paper reranks top-300 fused candidates
- Filter candidates below threshold tau=0.5

**Expected impact:** 0.45 → 0.55

---

### Stage 5: LLM Verification + Adaptive K (0.55 → 0.65+)

**Fix 5a: Qwen2.5-7B as verifier (NEVER generator)**
```python
verify_prompt = f"""You are a Swiss legal expert. Given the query and candidate citation,
score relevance from 0-10.

Query: {query_en}
Citation: {candidate_citation}
Document excerpt: {candidate_text[:500]}

Score (0-10):"""
score = int(qwen_generate(verify_prompt, max_tokens=5))
```
- Key principle: LLM SCORES candidates, never GENERATES citation strings
- Eliminates hallucinated citations (which score 0 in exact-match evaluation)
- Apply only to top-100 after cross-encoder reranking

**Fix 5b: Adaptive citation count**
```python
# Combine 3 signals for per-query K
n_pred = int(0.4 * n_llm + 0.3 * n_elbow + 0.3 * n_calibrated)
n_pred = max(8, min(50, n_pred))
```
- `n_llm`: Ask Qwen "How many citations would a legal expert provide for this query?"
- `n_elbow`: Find the score drop-off point in reranked candidates
- `n_calibrated`: Global K from val set grid search (our current approach)
- Clip to [8, 50] range

**Expected impact:** 0.55 → 0.65+

---

## Weighted RRF Signal Architecture (from reference paper)

| Signal | Weight | Source |
|--------|--------|--------|
| Cross-signal boost | 5.0 | Content ∩ Graph agreement |
| Citation seed expansion | 4.0 | Graph neighbor expansion |
| Direct citation lookup | 3.0 | Entity exact match |
| BM25 (German, LLM-translated) | 2.0 | Primary lexical |
| Sibling expansion | 1.5 | Same-parent court decision |
| Semantic (EN dense) | 1.2 | BGE-M3 cross-lingual |
| Semantic (DE dense) | 1.2 | BGE-M3 on translated query |
| BM25 (HyDE text) | 1.0 | Generated query variant |
| PPR | 0.8 | PageRank scores |
| Bibliographic coupling | 0.6 | Shared out-citations |
| HITS | 0.5 | Authority scores |
| Community (Leiden) | 0.3 | Leiden cluster members |

---

## Model Stack (Kaggle T4 x2, 15GB VRAM, sequential loading)

| Component | Model | Params | VRAM | Phase |
|-----------|-------|--------|------|-------|
| Embedding | BAAI/bge-m3 (fp16) | 568M | ~2GB | Stage 2 |
| Reranker | BAAI/bge-reranker-v2-m3 | 568M | ~2GB | Stage 4 |
| Translation+HyDE+Verify | Qwen2.5-7B (NF4 4-bit) | 7B | ~5GB | Stage 1,2,5 |
| BM25 | bm25s + PyStemmer | — | CPU | Stage 1 |
| Graph | scipy.sparse + PPR | — | CPU | Stage 3 |
| Vector Index | FAISS FlatIP (GPU) | — | ~1GB | Stage 2 |

Sequential loading: Qwen → unload → BGE-M3 → unload → BGE-reranker → unload → Qwen verify

---

## Implementation Priority

| Priority | What | Expected F1 Gain | Effort |
|----------|------|-------------------|--------|
| **P0** | Replace OpusMT with Qwen2.5-7B translation | 0.01 → 0.10+ | Medium |
| **P0** | Entity direct lookup (w=3.0) | +0.03-0.05 | Small |
| **P0** | Add PyStemmer to BM25 | +0.02-0.05 | Small |
| **P1** | HyDE query expansion | +0.05-0.10 | Medium |
| **P1** | BGE-M3 sparse mode | +0.01-0.03 | Small |
| **P1** | Citation graph + co-citation | +0.03-0.05 | Medium |
| **P2** | PPR on citation graph | +0.02-0.04 | Medium |
| **P2** | Cross-signal boosting (w=5.0) | +0.03-0.05 | Small |
| **P2** | BGE-reranker-v2-m3 | +0.05-0.07 | Medium |
| **P3** | LLM verification | +0.04-0.06 | Medium |
| **P3** | Adaptive citation count | +0.02-0.04 | Small |

---

## Kaggle Runtime Budget (12h limit)

| Stage | Estimated Time |
|-------|---------------|
| Qwen2.5-7B translate 50 queries | ~3 min |
| Qwen2.5-7B HyDE generate 50 passages | ~5 min |
| BGE-M3 encode 175K laws (dense+sparse) | ~25 min |
| FAISS index build | ~1 min |
| BM25 index build (175K laws + 1.6M court) | ~15 min |
| Citation graph construction (regex scan) | ~10 min |
| PPR computation (50 queries) | ~2 min |
| Retrieval (all signals, 50 queries) | ~5 min |
| BGE-reranker-v2 rerank top-300 x 50 queries | ~15 min |
| Qwen2.5-7B verify top-100 x 50 queries | ~30 min |
| Calibration + submission write | ~1 min |
| **Total** | **~112 min (~1.9 hours)** |

Well within 12-hour limit with 10+ hours safety margin.

---

## Key Insights from Papers

### From BGE-M3 paper:
- Sparse retrieval is WEAK cross-lingually (43.3 vs 76.2 dense on MKQA-de)
- BUT useful as supplement for same-language matching (translated queries)
- ColBERT multi-vector as reranker: use `compute_score(weights=[0.4, 0.2, 0.4])`
- For long docs: sparse weight should be higher (w2=0.8)

### From CLIR papers:
- OpusMT (134M) is catastrophically bad — BLEU=10.77 vs GPT-4o=34.41
- This is the PRIMARY root cause of our F1=0.01
- Dense embeddings WITHOUT translation outperform BM25+OpusMT
- HyDE: generate hypothetical docs in TARGET language (German)

### From Graph RAG papers:
- PPR alpha=0.5, max_iter=50 sufficient
- Node specificity = 1/|passages_containing_node| (IDF-like)
- Regex extraction sufficient for Swiss legal citations (no LLM needed)
- Use scipy sparse, not NetworkX, for 2.47M-node graphs
- RAG+KG = 70% vs RAG-only = 37.5% (Domain-Partitioned paper)

### From Swiss Legal NLP papers:
- SCALE benchmark: same research group behind competition data
- Swiss citation formats are highly structured and regex-parseable
- LEXam: Swiss law exam questions similar to competition queries
- Cumulative text units: prepend parent article text for sub-articles (improves embedding quality)

---

*Synthesized from 15 reference papers by 4 parallel research agents.*
*Next step: Rebuild notebook_kaggle.ipynb following this plan.*
