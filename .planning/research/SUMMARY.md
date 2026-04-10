# Project Research Summary

**Project:** LLM Agentic Legal Information Retrieval (Kaggle Competition)
**Domain:** Cross-lingual legal citation retrieval - English queries over German Swiss law corpus
**Researched:** 2026-04-10
**Confidence:** MEDIUM-HIGH

## Executive Summary

This is a Kaggle competition for cross-lingual retrieval of Swiss legal citations. Given English-language legal questions, the system must predict the exact set of citations (statutes and court decisions) that would appear in a legal opinion. The metric is Macro-F1, which punishes both under-prediction and over-prediction equally. The current baseline scores F1=0.009 due to three compounding failures: the 2.47M court decisions corpus is entirely excluded (missing 41% of gold citations by definition), GPU is not enabled (encoding takes hours on CPU instead of seconds), and a fixed count of 100 predictions is returned per query when the validation set expects approximately 25.

The recommended approach follows a six-stage sequential pipeline with two parallel retrieval sub-pipelines that converge at fusion: a full dense + sparse index for the 175K laws corpus, and BM25-only retrieval with on-the-fly dense reranking for the 2.47M court corpus (full dense indexing of court decisions is infeasible in a 12-hour runtime budget). BGE-M3 replaces multilingual-e5-large as the embedding model, bm25s replaces rank-bm25 for the court corpus, and a Jina multilingual cross-encoder handles final reranking. A 4-bit quantized Qwen2.5-7B LLM handles query expansion and per-query citation count estimation. All models are loaded sequentially, never simultaneously, to stay within the 16GB T4 VRAM budget.

The dominant risk is runtime failure: the 2.47M court BM25 index is the critical path and must use bm25s (not rank-bm25, which will OOM or exhaust 12 hours just during indexing). The second major risk is silent failures -- GPU not allocated, model not on GPU, wrong embedding prefixes -- any of which produce a wasted 12-hour run. Build order matters: fix GPU first, add court BM25 second, then entity/graph signals, then quality improvements. Each layer is independently submittable.

---

## Key Findings

### Recommended Stack

The stack is constrained by Kaggle offline environment (no internet, no external servers), a 12-hour runtime limit, and two T4 GPUs (16GB VRAM each, 30GB RAM). Every major technology choice is driven by these constraints. BGE-M3 (560M params) is the embedding model -- it outperforms multilingual-e5-large on MIRACL, supports 8192-token sequences for long Swiss legal statutes, and delivers dense+sparse retrieval from a single checkpoint. bm25s (not rank-bm25) is the only viable BM25 library for the 2.47M court corpus: 500x faster with memory-mapped index support. FAISS with faiss-gpu-cu12 handles the laws corpus dense index (175K documents fit in ~730MB VRAM). The LLM is Qwen2.5-7B-Instruct in 4-bit NF4 quantization (~4.5GB VRAM), loaded only during query analysis and unloaded before any other neural model is active.

**Core technologies:**
- BAAI/bge-m3: Primary embedding model -- outperforms e5-large on cross-lingual MIRACL, 8K context, dense+sparse retrieval from one checkpoint
- bm25s (0.2.x): BM25 for court corpus -- 500x faster than rank-bm25, memory-mapped index, handles 2.47M docs in 30-45 min
- jinaai/jina-reranker-v2-base-multilingual: Cross-encoder reranker -- 15x faster than bge-reranker, multilingual-trained
- Qwen/Qwen2.5-7B-Instruct (4-bit NF4): Query expansion + citation count estimation -- stronger German/multilingual than Mistral-7B, confirmed T4 compatibility
- faiss-gpu-cu12: GPU ANN index for laws corpus -- 10-50x faster than CPU FAISS, IndexFlatIP exact search fits in ~730MB VRAM
- Helsinki-NLP/opus-mt-tc-big-en-de: EN->DE translation for BM25 queries -- CPU-only; BGE-M3 handles dense retrieval cross-lingually without translation

### Expected Features

The baseline fails not from missing advanced features but from missing table-stakes coverage. Five must-have features must exist before any differentiator matters.

**Must have (table stakes):**
- BM25 over court corpus (bm25s) -- without this, 41% of gold citations are permanently unrecoverable
- GPU enablement + hard assertion -- CPU fallback silently consumes the entire 12-hour runtime
- EN->DE query translation -- BM25 requires German terms to match German corpus text
- Per-query citation count calibration -- returning 100 predictions when gold average is 25 destroys precision
- RRF fusion of BM25 + dense signals -- no single signal dominates; parameter-light and proven in legal retrieval competitions

**Should have (competitive differentiators):**
- Entity-driven direct lookup (Art. X / SR number regex) -- guaranteed precision for queries with explicit article references
- Citation graph expansion (law article to court decisions citing it) -- converts laws hits into court hits with no additional dense retrieval
- BGE-M3 as drop-in replacement for e5-large -- higher MIRACL score, 8K context, no added VRAM cost
- Dense reranking of BM25 court top-500 candidates -- semantic signal layered on top of BM25 recall
- Cross-encoder reranking on final merged top-150 -- COLIEE 2024/2025 winning pipeline step
- LLM-based HyDE German query expansion -- 3-17% nDCG improvement on out-of-domain legal corpora
- Score-based fusion weights learned from val set -- validated per-signal weights replace equal-weight RRF

**Defer indefinitely:**
- Full dense encoding of 2.47M court corpus -- approximately 8 hours embedding time, exceeds budget
- Any embedding model fine-tuning -- requires days; infeasible in 12-hour limit
- FR/IT court decision retrieval -- effort not justified given German-heavy val/test composition
- LLM-generated citation lists -- hallucinated citations score 0; Qwen/Mistral have no Swiss law training

### Architecture Approach

The architecture is a six-stage sequential pipeline: data loading, index building (BM25 laws, BM25 court, FAISS laws dense, entity lookup table), per-query processing (entity extraction, translation, optional LLM expansion), retrieval (laws sub-pipeline: BM25+dense; court sub-pipeline: BM25 top-500 then on-the-fly dense rerank), RRF fusion + cross-encoder reranking, optional LLM augmentation for count estimation, and calibration + output. Components are stateless per-query; shared state is only read-only index objects. Models are passed as constructor arguments to make sequential lifecycle explicit.

**Major components:**
1. BM25IndexBuilder -- builds bm25s indices for both corpora; court index is the longest-running step (~30-45 min)
2. FAISSIndexBuilder -- embeds 175K laws with BGE-M3 on GPU, builds IndexFlatIP; immutable after build
3. EntityLookupBuilder -- builds (law_code, art_num) to doc_ids dict and citation graph dict; O(1) per-query lookup after build
4. QueryAnalyser -- regex entity extraction + OpusMT translation + optional Qwen2.5 expansion; stateless per query
5. RRFFuser -- merges BM25-laws, dense-laws, BM25-court, entity-lookup ranked lists; weighted RRF preferred over equal weights
6. CrossEncoderReranker -- jina-reranker-v2 scores top-200 fused candidates; batched 32 pairs per forward pass
7. MistralAugmentor (optional) -- Qwen2.5-7B 4-bit, loaded only during Stage 5, unloaded after; gated by USE_LLM_AUGMENTATION flag
8. Calibrator -- grid-searches K on val set only; applies learned K to test; never calibrates on test set

### Critical Pitfalls

1. Silent CPU fallback wastes entire run (AH-1) -- add assert torch.cuda.is_available() as the first notebook cell; log device name; set a 60-second encoding checkpoint
2. rank-bm25 OOMs or times out on 2.47M docs (CP-5) -- use bm25s exclusively for court corpus; filter to German-only documents first; build index in batches of 500K if RAM is tight
3. bfloat16 on T4 causes silent garbage output (CP-3) -- always use bnb_4bit_compute_dtype=torch.float16 and attn_implementation=eager when loading Qwen/Mistral on T4; Flash Attention 2 requires compute capability >=8.0 (T4 is 7.5)
4. Swiss citation format mismatches cause false negatives (CP-6) -- build canonicalization (normalize whitespace, Swiss ss convention, Roman numerals); fuzzy-snap LLM-generated citations to corpus canonical form using Levenshtein distance <=2
5. Macro-F1 calibration overfitting on 10 val queries (CP-4 / AH-5) -- train/val citation count distributions differ 6x (4.1 vs 25.1 mean); calibrate on val only; use per-query LLM count estimate rather than a single global K

---

## Implications for Roadmap

Five phases are recommended. Each phase is independently submittable and builds strictly on the previous. The ordering follows dependency flow and is validated by the FEATURES.md MVP recommendation.

### Phase 1: Foundation -- GPU + Laws Pipeline
**Rationale:** The existing code almost works for the laws corpus (59% of gold citations) but is broken by CPU-forced inference. Fixing GPU enablement requires zero new models and costs hours of wasted runtime if not done first. Establishes the F1 baseline for laws-only retrieval.
**Delivers:** Working GPU-enabled pipeline; laws BM25 + BGE-M3 dense retrieval; calibrated K; valid submission
**Addresses:** GPU enablement (table stakes), e5-large to BGE-M3 upgrade, cross-encoder reranking on laws candidates, citation count calibration
**Avoids:** AH-1 (silent CPU fallback), AH-4 (wrong model size), MP-6 (missing e5 prefix assertion), MP-5 (CSV format rejection)

### Phase 2: Court Corpus Coverage
**Rationale:** 41% of gold citations are court decisions and the current pipeline returns 0% recall on them. This single gap explains most of the score deficit. BM25 is the only feasible full-corpus signal at 2.47M documents; adding it requires no new GPU-resident models.
**Delivers:** BM25 retrieval over 2.47M court decisions; RRF fusion of laws + court signals; expected F1 jump to 0.08-0.15
**Uses:** bm25s (replacing rank-bm25), OpusMT EN->DE translation for BM25 query term matching
**Implements:** Court BM25 sub-pipeline; weighted RRF with corpus-specific signal weights
**Avoids:** CP-5 (rank-bm25 OOM), CP-1 (German compound blindness via decompounding + stopwords), CP-7 (equal-weight RRF diluting precision), MN-1 (German-only filter reduces index size and noise)

### Phase 3: Entity + Citation Graph Signals
**Rationale:** Many legal queries contain explicit article references that can be looked up with perfect precision. The citation graph converts a laws hit into court decision hits without additional dense retrieval, directly amplifying court coverage from Phase 2. Both require no additional models.
**Delivers:** Entity extraction (regex), direct statute lookup, citation graph expansion (law to court decisions citing it), nearby article expansion
**Implements:** EntityLookupBuilder, EntityRetriever components; citation format canonicalization
**Avoids:** CP-6 (citation format normalization built here), MP-4 (empty RRF guards when entity signal absent)

### Phase 4: Quality Signals -- Dense Court Reranking + Advanced Fusion
**Rationale:** After BM25 court retrieval is working, semantic quality of court candidates improves by embedding BM25 top-500 on-the-fly and reranking with the dense model. Score-based fusion weights learned on val replace equal RRF weights.
**Delivers:** On-the-fly dense reranking of BM25 court top-500; validation-fitted fusion weights; cross-encoder on merged top-150; expected +0.05-0.08 F1
**Uses:** BGE-M3 for on-the-fly court candidate embedding; jina-reranker-v2 for cross-encoder; grid search or Optuna for fusion weights
**Implements:** DenseRetriever (court on-the-fly path); CrossEncoderReranker with batched inference; score-based RRF weighting
**Avoids:** CP-8 (cross-encoder latency -- cap at 200 candidates, batch 32 pairs per pass, truncate to 256 tokens), MP-1 (no full dense index for court corpus)

### Phase 5: LLM Augmentation
**Rationale:** Qwen2.5-7B in 4-bit NF4 enables HyDE-style German query expansion and per-query citation count estimation -- both address calibration weaknesses from the small val set. Highest-risk phase; must remain optional.
**Delivers:** German query expansion (HyDE); per-query K estimation blended with calibrated K; expected +0.03-0.07 F1 with high variance
**Uses:** Qwen2.5-7B-Instruct 4-bit NF4, bitsandbytes, sequential load/unload lifecycle
**Implements:** MistralAugmentor gated by USE_LLM_AUGMENTATION flag
**Avoids:** CP-3 (bfloat16/Flash Attention T4 incompatibility), CP-4 (overfitting calibration), hallucination anti-feature (never generate citation lists directly from LLM)

### Phase Ordering Rationale

- GPU first because every subsequent phase depends on neural inference; broken GPU makes all phases impossible to validate
- Laws pipeline before court because 59% of citations are laws -- laws-only submission validates the full pipeline end-to-end
- Court BM25 before entity/graph because the citation graph needs BM25-identified court doc IDs as input
- Quality signals before LLM because dense reranking and cross-encoder have lower implementation risk and directly proven F1 gains
- LLM last because it is optional, high-variance, and all prior layers run without it

### Research Flags

Phases likely needing deeper targeted research during planning:
- Phase 3 (Citation Graph): Implementation details for building the court-document to law-article inverted index are sparse in published research. May need targeted research on Swiss citation format parsing.
- Phase 5 (LLM Augmentation): LLM citation count estimation has no published benchmarks for this specific domain. Measure val F1 impact on Phase 4 submission before committing Phase 5 runtime.

Phases with standard patterns (skip research-phase):
- Phase 1: Well-documented BGE-M3 + FAISS GPU patterns; stack confirmed in official model cards and COLIEE results
- Phase 2: bm25s fully documented; German BM25 tokenization patterns established in GerDaLIR literature
- Phase 4: Cross-encoder reranking pipeline is standard; patterns from COLIEE 2024/2025 well-documented

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | BGE-M3, bm25s, jina-reranker-v2, faiss-gpu-cu12 confirmed via official model cards, PyPI benchmarks, and MIRACL/COLIEE results. VRAM figures are MEDIUM -- runtime peaks depend on KV-cache and activation memory |
| Features | HIGH for table stakes; MEDIUM for differentiator F1 estimates | 41% court gap confirmed from competition data. Differentiator estimates extrapolated from COLIEE 2024/2025, not this dataset specifically |
| Architecture | HIGH for component design; MEDIUM for time estimates | Six-stage pipeline is clean and research-validated. BM25 court index build time (30-45 min) has LOW confidence |
| Pitfalls | HIGH for team-confirmed (AH-1 to AH-5); MEDIUM-HIGH for research-verified (CP-1 to CP-8) | T4 bfloat16 incompatibility and BM25 compound word blindness confirmed by primary sources |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- BM25 court index RAM usage: bm25s sparse matrix for 2.47M docs estimated at 3-8GB RAM but not confirmed for this corpus vocabulary size. Fallback: chunked TF-IDF prefiltering (sklearn) to reduce court candidates before BM25 indexing.
- German decompounding library choice: Charbert tokenizer and iknow decompounding dict mentioned in PITFALLS but not benchmarked for this corpus. If BM25 court recall is poor, dedicated investigation needed.
- Citation graph construction speed: Scanning 2.47M court texts to build the inverted index is estimated at 5-10 min but unconfirmed -- could be a Phase 3 bottleneck.
- LLM citation count estimation quality: No published evidence exists for this technique on Swiss legal data. Treat as experimental; measure val delta before relying on it for test submissions.

---

## Sources

### Primary (HIGH confidence)
- BGE-M3 model card and MIRACL benchmarks: https://huggingface.co/BAAI/bge-m3 and https://arxiv.org/abs/2402.03216
- bm25s paper and benchmarks: https://arxiv.org/abs/2407.03618 and https://bm25s.github.io/
- Jina Reranker v2 model card: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
- COLIEE 2025 (UQLegalAI with BGE-M3): https://arxiv.org/html/2505.20743v1
- bitsandbytes T4 compute capability: https://github.com/TimDettmers/bitsandbytes/issues/529
- FAISS GPU documentation: https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/
- GerDaLIR (German legal IR, BM25 stemming guidance): https://github.com/lavis-nlp/GerDaLIR

### Secondary (MEDIUM confidence)
- Qwen2.5-7B T4 inference benchmark: https://medium.com/@wltsankalpa/benchmarking-qwen-models-across-nvidia-gpus-t4-l4-h100-architectures-finding-your-sweet-spot-a59a0adf9043
- Hybrid GraphRAG for Swiss legal citation retrieval: https://www.ijecs.in/index.php/ijecs/article/view/5461
- HyDE cross-lingual query expansion: https://arxiv.org/html/2511.19325v1
- Weighted RRF in practice: https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf
- Macro-F1 pitfalls with small val sets: https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/

### Tertiary (LOW confidence)
- LLM citation count estimation: logical extension of calibration work -- no direct evidence found for this technique
- opus-mt-tc-big-en-de quality on legal domain: model card only; no legal benchmark data found
- BM25 court index RAM estimate for 2.47M docs: extrapolated from bm25s 2M-doc NQ example, not confirmed for Swiss legal corpus vocabulary

---
*Research completed: 2026-04-10*
*Ready for roadmap: yes*
