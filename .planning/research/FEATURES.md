# Feature Landscape: LLM Agentic Legal Information Retrieval

**Domain:** Cross-lingual legal citation retrieval (English queries → German Swiss law corpus)
**Competition:** Kaggle "LLM Agentic Legal Information Retrieval" — Macro-F1 metric
**Researched:** 2026-04-10
**Current baseline:** Macro-F1 = 0.009 (dense-only, e5-small, laws-only, CPU-forced)

---

## Context: Why the Baseline Fails

Before categorizing features, the root causes of F1=0.009 must be named:

1. **Zero court decision coverage** — 41% of gold citations are court decisions (BGE/docket numbers), system predicts 0%. This single gap explains most of the score gap.
2. **Citation count mismatch** — Val expects ~25 citations/query, system returns a fixed 100 (F1 is symmetric — over-prediction and under-prediction both hurt).
3. **CPU-only inference** — multilingual-e5-small on CPU for dense retrieval; too slow to do anything useful.
4. **Corpus limitation** — laws corpus (175K docs) only, missing the 2.47M court decisions corpus entirely.

Any feature that does not address at least one of these four root causes should be deprioritized.

---

## Table Stakes

Features where absence means the system cannot compete. These must exist before any differentiator matters.

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| **BM25 over court corpus** | Without this, 0% court recall. BM25 is the only feasible full-corpus signal over 2.47M docs. Dense encoding of 2.47M is infeasible in 12hr. | Medium | Use bm25s (not rank-bm25) — 500x faster, memory-mapped, handles 2M+ docs. Index build: ~15–30 min for 2.47M docs. |
| **GPU enablement in Kaggle notebook** | Everything downstream (dense retrieval, reranking, LLM) requires GPU. CPU e5-small is the primary reason submission underperforms local solution. | Low | Fix accelerator selection in notebook; confirm T4 is allocated before model loads. |
| **Cross-lingual query translation (EN→DE)** | Entire corpus is German. Queries are English. Without translation, BM25 and sparse signals fail completely. | Low | OpusMT exists in solution.py. NLLB-200 is an alternative with broader language coverage. LLM-based translation (Mistral) likely higher quality but slower. |
| **Per-query citation count calibration** | Val expects ~25 citations; returning 100 scores ~0.25 precision at best. Returning 5 scores ~0.2 recall at best. F1 requires both to be non-zero. The optimal count varies per query. | Medium | Simple approach: fit a scalar offset on val set. Advanced: LLM estimates count per query. Either beats fixed-k. |
| **RRF fusion of BM25 + dense signals** | No single signal dominates. BM25 catches lexical matches (statute numbers), dense catches semantic matches (legal concepts). RRF is parameter-light and robust — proven in legal retrieval competitions. | Low | Already in solution.py. Apply same pattern to court corpus. k=60 default is solid; minor tuning via val set. |

---

## Differentiators

Features that lift score above "any team using the standard stack." Each provides independent F1 gain.

| Feature | Value Proposition | Complexity | Dependency | Notes |
|---------|-------------------|------------|------------|-------|
| **Entity-driven direct lookup** | Parse explicit "Art. X OR/SR Y.Z" references from query text. Exact-match lookup gets perfect precision for those citations. Contributes high-quality seeds before any retrieval. | Medium | Requires statute corpus with normalized IDs | Regex extraction of "Art.\s*\d+" and law codes (SR numbers, OR/ZGB/SchKG abbreviations) from English queries. Bypasses retrieval entirely for explicit citations — guaranteed precision. |
| **Citation graph expansion** | If statute Art. X is found, find BGE decisions that cite Art. X in their considerations. Converts a laws hit into court hits without additional dense retrieval. Directly addresses the 41% court gap. | High | Requires BM25 over court corpus + entity lookup | court_considerations.csv contains the text of decisions. Pre-build an inverted index: article_id → list of decision_ids that mention it. Lookup is O(1) per article. |
| **Nearby-article expansion (same law)** | For statutes: if Art. 28 OR is found, include Art. 27, 29 in candidates. Legal questions often cite adjacent provisions. Boosts recall at low cost. | Low | Requires structured law ID parsing | Parse SR number + article number; add ±2 neighbors. Downrank with RRF weight < 1.0 vs direct hits. |
| **Dense reranking of BM25 court candidates** | BM25 retrieves top-K court candidates fast; dense model scores them for precision. Combines scale of BM25 with quality of dense retrieval. | Medium | GPU + BM25 court retrieval | Encode top-200 BM25 court results with e5-large (batch on GPU). Rerank. Do not encode full 2.47M. |
| **BGE-M3 as drop-in replacement for e5-large** | BGE-M3 outperforms multilingual-e5-large on MIRACL (67.8 vs 65.4 nDCG@10), supports dense+sparse+colbert heads simultaneously, 8192-token context. Zero extra cost if swapped in. | Low | GPU only | BGE-M3 is available on HuggingFace. Fits in T4 VRAM (570M params ≈ ~2.2GB fp16). COLIEE 2025 top systems used BGE-M3. |
| **LLM-based German query expansion (HyDE-style)** | Mistral-7B generates a hypothetical German legal document answering the query, then that document is used as the retrieval query. Closes the vocabulary mismatch between English legal questions and German statutory text beyond what translation alone achieves. | High | GPU + Mistral 4-bit loaded | Generate 2–3 hypothetical German passages per query. Average their embeddings. Combine with translated-query embedding via RRF. Published results show 3–17% nDCG improvement on out-of-domain legal corpora. |
| **LLM-estimated per-query citation count** | Ask Mistral: "How many legal citations would you expect for this question?" Answers like "this involves a single contract provision" vs "this is a multi-statute tax dispute" predict citation volume. More accurate than a global calibration scalar. | High | Mistral loaded + calibration baseline working | Use as a multiplier on the calibrated baseline count. Cap at validation distribution (≤50). |
| **Cross-encoder reranking on final candidate set** | mMiniLM (or BGE-reranker-base) reranks the merged top-K candidates from all signals before outputting predictions. State-of-the-art step in all top legal IR systems (COLIEE 2024/2025 winners used this). Precision improvement at near-zero recall cost. | Medium | All retrieval signals merged first | Already in solution.py. Port to Kaggle notebook. Apply after RRF fusion, before calibration. Limit input to top-150 candidates to fit runtime. |
| **Score-based fusion weighted by validation F1** | Replace equal-weight RRF with learned weights (one float per signal) fit on 10 val queries. Convex combination of BM25-laws, BM25-court, dense-laws, dense-court, entity-lookup signals. Even tiny val set gives better weights than uniform. | Medium | All signals available | Simple grid search or Optuna over 5 float weights. Objective: val Macro-F1. This is the final calibration layer. |

---

## Anti-Features

Things to deliberately not build. Each has a specific reason grounded in constraints.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full dense encoding of 2.47M court corpus** | At ~100 docs/sec on GPU, encoding 2.47M would take ~7 hours — more than half the 12-hour budget, leaving no time for retrieval or reranking. | BM25 retrieval → top-200 candidates → dense rerank those 200 only. |
| **Fine-tuning embedding models within Kaggle** | Training e5-large or BGE-M3 on domain data requires days, not hours, even with LoRA. Infeasible within 12-hour runtime. | Use BGE-M3 pretrained. It already covers German legal domain through multilingual pretraining. |
| **Pre-uploaded Kaggle dataset (dense index)** | The project constraint is Kaggle-only. Pre-uploading a FAISS index would simplify things but is out of scope by explicit decision. | BM25 for court corpus (builds fast), dense for laws corpus (small enough to encode at startup). |
| **FR/IT court decisions retrieval** | court_considerations.csv is mixed DE/FR/IT. French and Italian processing requires additional language detection, tokenization, and translation overhead. The majority of val/test citations will be German-language decisions. | Filter to German-language court rows at index time using a fast language detector (langdetect or simple heuristic on character set). |
| **Full LLM-generated citation list (hallucination-based)** | Asking Mistral "which statutes apply?" and using those as predictions is seductive but unreliable. Mistral was not trained on Swiss law citation patterns; hallucinated citations score 0 and cost F1. | Use LLM only for query expansion and count estimation. All final predictions must come from corpus retrieval. |
| **Re-ranking the entire court corpus with a cross-encoder** | Cross-encoding 2.47M doc-query pairs is O(n) inference — completely infeasible regardless of hardware. | Cross-encoder sees only top-150 after BM25+dense fusion. |
| **Elastic / Solr / external search engine** | No internet access in Kaggle offline notebooks. External services are unavailable. | BM25s (pure Python, in-memory or memory-mapped) is the correct tool. |
| **Contrastive fine-tuning via negatives sampling** | Requires labeled pairs and multi-hour training. This competition has no labeled negative pairs and 12-hour time constraint. | Use pretrained BGE-M3 which already embeds contrastive training from multilingual data. |

---

## Feature Dependencies

```
GPU enablement
    └── Dense laws retrieval (e5-large / BGE-M3)
    └── Cross-encoder reranking
    └── Mistral 4-bit inference
          └── LLM query expansion (HyDE-style)
          └── LLM-estimated citation count

EN→DE translation
    └── BM25 court retrieval (German query needed)
    └── Dense retrieval (German query needed)

BM25 court retrieval
    └── Dense reranking of court candidates
    └── Citation graph expansion (needs court doc IDs)

Entity extraction (Art. X / SR numbers)
    └── Direct lookup (exact match on laws)
    └── Citation graph expansion (seeds the graph walk)

All retrieval signals present
    └── RRF fusion
          └── Cross-encoder reranking
                └── Per-query citation calibration
                      └── Final submission output
```

---

## MVP Recommendation

Ranked by F1 impact / implementation risk ratio. Each phase is independently submittable.

**Phase 1 — Court corpus coverage (highest impact, medium effort)**
1. Fix GPU enablement
2. Add BM25 court retrieval (bm25s, memory-mapped)
3. Translate queries EN→DE, run BM25 on court corpus
4. RRF-merge laws results + court results
5. Fix citation count calibration (target 20–30)

Expected: F1 jump from ~0.009 to 0.08–0.15 (estimated, based on 41% court coverage gap)

**Phase 2 — Entity + graph signals (high precision uplift)**
1. Entity extraction (Art. numbers, SR codes from query)
2. Direct statute lookup (exact match)
3. Citation graph expansion (laws → court decisions citing those laws)

Expected: +0.05–0.10 additional F1

**Phase 3 — Quality signals (precision/recall balance)**
1. Upgrade to BGE-M3 (replaces e5-large)
2. Dense reranking of BM25 court candidates
3. Cross-encoder reranking on merged top-150
4. Score-based fusion weights from val set

Expected: +0.05–0.08 additional F1

**Phase 4 — LLM augmentation (differentiating, high effort)**
1. Mistral 4-bit: HyDE-style German query expansion
2. Mistral 4-bit: per-query citation count estimation
3. VRAM lifecycle management (load/unload sequentially)

Expected: +0.03–0.07 additional F1, high variance

**Defer indefinitely:**
- Citation graph analysis beyond direct article→decision mapping (Personalized PageRank, co-citation) — complexity high, marginal gain at small scale
- Multilingual court filtering (FR/IT) — effort vs. German-only recall not justified given test set composition
- Any model fine-tuning

---

## Confidence Assessment

| Feature Area | Confidence | Basis |
|---|---|---|
| BM25 for large court corpus | HIGH | bm25s documentation, COLIEE competition outcomes, tested on 2M+ doc NQ dataset |
| RRF fusion effectiveness | HIGH | COLIEE 2024/2025 winners all used it; published results consistent |
| BGE-M3 > multilingual-e5-large | HIGH | MIRACL benchmark, COLIEE 2025 UQLegalAI results with BGE-M3 |
| Citation graph expansion | MEDIUM | Direct evidence from GraphRAG paper on Swiss legal system specifically; implementation details sparse |
| HyDE / LLM query expansion | MEDIUM | Published gains of 3–17% nDCG on out-of-domain; German legal domain specifically not benchmarked |
| LLM citation count estimation | LOW | No direct evidence for this specific technique; logical extension of calibration work |
| Score-based fusion on 10 val queries | MEDIUM | Standard practice; 10-query val set makes overfitting a real risk |

---

## Sources

- COLIEE 2025 overview: https://dl.acm.org/doi/10.1145/3769126.3785016
- UQLegalAI@COLIEE2025 (graph + BGE-M3): https://arxiv.org/html/2505.20743v1
- BGE-M3 paper (MIRACL benchmarks): https://arxiv.org/abs/2402.03216
- Hybrid GraphRAG for Swiss legal citation retrieval: https://www.ijecs.in/index.php/ijecs/article/view/5461
- Generative query expansion cross-lingual IR: https://arxiv.org/html/2511.19325v1
- bm25s (fast BM25 for large corpora): https://github.com/xhluca/bm25s
- GuRE legal query rewriting: https://aclanthology.org/2025.nllp-1.31.pdf
- LeCNet citation network benchmark: https://aclanthology.org/2025.justnlp-main.4.pdf
- Swiss NLP benchmark (multilingual): https://arxiv.org/abs/2410.13456
- RRF hybrid search: https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/
- AusLaw Citation Benchmark (LLM citation prediction): https://arxiv.org/html/2412.06272
