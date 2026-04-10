# Domain Pitfalls: LLM Agentic Legal Information Retrieval

**Domain:** Kaggle competitive retrieval — cross-lingual legal citation retrieval (en→de, Swiss law)
**Researched:** 2026-04-10
**Confidence:** HIGH for pitfalls already hit by this team; MEDIUM for verified-from-research pitfalls; LOW flagged individually

---

## Already-Hit Pitfalls (Confirmed Expensive)

These are documented failures from the team's own history. They are stated here as prevention guides for recurrence.

### AH-1: CPU Fallback Silent Failure (1.7hr encoding time)

**What goes wrong:** When `torch.cuda.is_available()` returns False on Kaggle (e.g., due to a CUDA driver mismatch or notebook restart without GPU kernel), the code silently falls back to CPU. Dense encoding of the laws corpus alone takes ~1.7 hours on CPU. The 2.47M court corpus would take over 24 hours — consuming the entire 12-hour Kaggle budget before inference begins.

**Why it happens:** The Kaggle T4 GPU kernel must be explicitly selected at notebook creation time. Restarting a notebook or copying it can default back to CPU. Additionally, CUDA import errors from library version mismatches (e.g., bitsandbytes compiled against wrong libcudart) can cause `cuda.is_available()` to return False even when the GPU is physically present.

**Consequences:** Entire 12-hour run consumed with no valid submission. Undetected until hours in.

**Warning signs:**
- `torch.cuda.is_available()` returns False at notebook start
- `device = 'cpu'` logged
- Encode batch completes in >5s per batch of 32 (GPU should do 32 docs in <0.5s)
- No GPU utilization shown in Kaggle sidebar

**Prevention:**
- Add hard assert at cell 1: `assert torch.cuda.is_available(), "GPU not available — abort"` — fail immediately, not silently
- Log device at startup with `print(f"Device: {torch.cuda.get_device_name(0)}")`
- Set a 60-second wall-clock checkpoint after encoding begins; if <1000 docs encoded, abort

**Phase:** GPU setup validation cell — must be the first executable cell in the notebook

---

### AH-2: Missing Court Corpus (41% of Gold Citations Unrecoverable)

**What went wrong:** The pipeline only retrieved from laws_de.csv (175K rows). Court decisions (BGE/BGer dockets) in court_considerations.csv (2.47M rows) were entirely excluded. Validation gold citations are 41% court decisions — meaning the maximum achievable recall without court retrieval is ~59%.

**Warning signs:**
- Val Macro-F1 stays below 0.10 even after tuning other parameters
- Predicted citations are all "Art. X OR" style, never "BGE X Y Z" or docket format
- Running `gold_citations.str.contains('BGE').mean()` on val set returns >0.3

**Prevention:**
- Run corpus coverage audit before any tuning: count what fraction of gold citations are findable in each sub-corpus
- Treat court corpus as a first-class retrieval source, not an afterthought
- Any pipeline evaluation must report per-corpus recall separately

**Phase:** Corpus audit — must run before any retrieval component is built

---

### AH-3: Fixed Prediction Count Destroying Precision

**What went wrong:** Hardcoding 100 predictions per query produces near-zero precision when gold sets average 25 citations. Macro-F1 penalizes both FP and FN equally per query, so flooding predictions with 100 items where only 25 are correct gives precision=0.25, recall=1.0 at best, F1≈0.40 per query — but in practice recall was never 1.0, making it worse.

**Warning signs:**
- Submission has exactly 100 semicolon-separated citations for every query
- Precision metric (if tracked) is very low (<0.3) while recall is high
- F1 does not improve when recall improves further

**Prevention:**
- Calibrate per-query prediction count on val set using binary search over threshold
- Train a count predictor using query features (length, entity count, specificity)
- Use Mistral-7B to estimate "how many laws does this question implicate?"
- Never use a fixed count; query complexity varies from 1 citation to 50+

**Phase:** Calibration layer — after retrieval is functional, before submission generation

---

### AH-4: e5-small vs e5-large (Near-Zero Recall)

**What went wrong:** multilingual-e5-small has 117M parameters vs e5-large at 560M. On cross-lingual legal retrieval (en query → de corpus), the small model's embedding space is insufficiently aligned — legal terminology in particular requires the richer multilingual representation in the large model. Recall went from near-zero to meaningful only after upgrading.

**Warning signs:**
- Dense retrieval recall@100 < 5% on val set
- BM25 outperforms dense retrieval by a large margin (BM25 should complement, not dominate)
- Dense index build time is suspiciously fast (<10 min for 175K docs on GPU)

**Prevention:**
- Always use multilingual-e5-large for cross-lingual legal retrieval — not e5-small, not e5-base
- Verify model name in code with `model.name_or_path` at load time
- Size check: e5-large embedding dimension is 1024, e5-small is 384 — assert dim==1024 after first encode

**Phase:** Model selection — enforced at notebook start, not changeable mid-run

---

### AH-5: Train/Val Distribution Mismatch Not Accounted For

**What went wrong:** Train queries average 4.1 citations per query; val queries average 25.1 per query (6x higher). A model calibrated on train data predicts too few citations for val queries, capping recall before it can improve F1.

**Warning signs:**
- Val Macro-F1 is much lower than train Macro-F1 with identical settings
- Calibration on train suggests K=5 as best, but val needs K=20-30
- Gold citation count distribution plots are bimodal

**Prevention:**
- Always inspect gold citation count distributions for train and val separately before calibrating
- Treat the 10 val queries as the only reliable calibration signal for the test distribution
- Cross-validate calibration on train using only queries with >10 gold citations as a proxy

**Phase:** Data exploration — before any modeling decisions

---

## Critical Pitfalls (High Impact, Research-Verified)

### CP-1: BM25 German Compound Word Blindness

**What goes wrong:** Standard BM25 tokenization (whitespace/punctuation split) treats German compound words as atomic tokens. "Bundesgesetz" does not match queries about "Gesetz", "Schweizerisches Obligationenrecht" does not match a query fragment containing just "Obligationenrecht". Since Swiss legal German uses heavy compounding (e.g., "Datenschutzgesetz", "Gerichtsstandsgesetz"), lexical matching fails for any query fragment that is itself part of a compound.

**Root cause:** BM25 with rank-bm25's default tokenizer has no compound splitting (decompounding). It also has no stopword removal by default — legal boilerplate ("der", "die", "das", "gemäß") pollutes IDF scores. Research confirms: for German legal retrieval, no stemming is better than aggressive stemming (Okapi BM25 loses precision with radicalization), but decompounding improves recall.

**Consequences:** BM25 misses relevant laws because the exact compound appears differently than in the query translation.

**Warning signs:**
- High exact-match queries retrieve correctly but paraphrase/fragment queries retrieve nothing
- Adding OR-style variants manually improves BM25 recall dramatically
- Laws with compound names consistently rank lower than their relevance warrants

**Prevention:**
- Apply German decompounding before BM25 indexing: use `charbert` tokenizer or `iknow` decompounding dict
- Swiss-specific: replace `ß` with `ss` in all tokens (Swiss German standard)
- Remove German stopwords explicitly before BM25 indexing
- Do NOT apply aggressive stemming to BM25 for German legal text — it hurts precision
- Use BM25S instead of rank-bm25: BM25S uses scipy sparse matrices and is 100-500x faster on large corpora, making it feasible for the 2.47M court corpus

**Phase:** BM25 implementation phase

---

### CP-2: Translation Quality Failures for Legal Queries (OpusMT)

**What goes wrong:** OpusMT (Helsinki-NLP/opus-mt-en-de) is a general-domain translation model. Legal English queries contain domain-specific terms ("tortious liability", "cantonal administrative proceeding", "enforcement of provisional measures") that OpusMT translates using general-purpose approximations, sometimes producing grammatically valid but legally wrong German phrases. The translated query then misses the exact German legal terminology used in the corpus.

**Research finding:** Neural MT models including OPUS-MT exhibit hallucinations — producing target-language output that is fluent but unrelated to the source meaning. In legal contexts this is particularly dangerous because precise term matching matters.

**Consequences:** A well-translated query retrieves relevant documents; a mistranslated query retrieves noise. Since all val/test queries are English and the corpus is German, translation quality directly gates retrieval quality.

**Warning signs:**
- Manual inspection reveals translated legal terms that differ from corpus terminology
- BM25 on translated query retrieves fewer results than BM25 on back-translated corpus text
- Queries with specialized legal terms (e.g., "sequestration", "proportionality principle") translate to generic German

**Prevention:**
- Use translation as one signal, not the only signal: always retrieve on both original English (for dense/multilingual models) and translated German (for BM25)
- Build a Swiss legal terminology dictionary: map common English legal terms to their canonical German Swiss equivalents (e.g., "OR" for Obligationenrecht)
- Add entity-first retrieval: if the query contains "Art. X OR" already, use it directly without translation dependency
- Consider query expansion in German using Mistral-7B to generate alternative phrasings

**Phase:** Translation and query expansion phase

---

### CP-3: VRAM Management — Sequential Model Loading Required

**What goes wrong:** Loading all three models simultaneously (multilingual-e5-large ~560MB VRAM, mMiniLM cross-encoder ~120MB, OpusMT ~300MB, Mistral-7B 4-bit ~4GB) plus FAISS index (laws: 175K × 1024 × 4 bytes ≈ 700MB) and court BM25 sparse matrix in RAM pushes total memory beyond what fits comfortably in 16GB T4 VRAM. Adding Mistral-7B requires the other models to be unloaded first.

**Research finding:** T4 GPU has compute capability 7.5. bitsandbytes 4-bit quantization is supported at 7.5 but borderline — the library warns "Only slow 8-bit matmul is supported for your GPU" for capability <7.5 but T4 is exactly 7.5. In practice, 4-bit NF4 (not int4) works on T4 but requires `bnb_4bit_compute_dtype=torch.float16` — using bfloat16 on T4 silently produces garbage outputs or crashes because T4 does not support bfloat16.

**Flash Attention:** Flash Attention V2 is NOT supported on T4 (requires compute capability >=8.0). Importing it will cause `CUDA error: no kernel image is available for execution on the device`. Must explicitly pass `attn_implementation="eager"` when loading Mistral on T4.

**Consequences:** OOM crash mid-run, or 12-hour timeout with no submission.

**Warning signs:**
- `RuntimeError: CUDA out of memory` during model loading
- `UserWarning: bfloat16 is not supported on T4` — often swallowed, causes silent quality degradation
- Process killed without error (OOM killer triggered at OS level)

**Prevention:**
- Explicit lifecycle management: load model → use → `del model; torch.cuda.empty_cache(); gc.collect()` → load next model
- Load order: Mistral-7B for query analysis → del → load e5-large for encoding → del → load cross-encoder for reranking
- For Mistral-7B on T4: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")` — not bfloat16
- Always pass `attn_implementation="eager"` to avoid Flash Attention import on T4
- Add VRAM monitoring: `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` after each model load
- Use `device_map="auto"` to split across both T4s when loading Mistral

**Phase:** Model lifecycle management — implement in the first notebook cell that loads any model

---

### CP-4: Macro-F1 Calibration Overfitting on 10 Val Queries

**What goes wrong:** The team calibrates prediction count threshold on 10 validation queries. With only 10 data points, the "best K" found by grid search overfits to val — particularly because val Macro-F1 is non-smooth: adding or removing one correct citation from one query changes the metric by 10% (1/10 of queries). The optimal K for val may be wrong for the 40-query test set.

**Research finding:** Macro-F1 is not invariant to prevalence shifts. If the test set has a different citation-count distribution than val (likely given train has 4.1 mean and val has 25.1 mean), the calibrated K will be wrong. Optimizing global threshold for macro-F1 also risks pathological all-positive behavior for rare-class queries.

**Consequences:** A pipeline that achieves 0.25 val Macro-F1 with K=25 may achieve 0.12 test Macro-F1 if the test distribution differs.

**Warning signs:**
- Best K found on val is at the boundary of the search range (e.g., K=1 or K=100)
- Val Macro-F1 jumps discontinuously with small K changes
- Per-query F1 variance is very high (some queries at 0.8, others at 0.0)

**Prevention:**
- Do NOT use a single global K — calibrate K per query using query features (entity count, query length, specificity score)
- Cross-validate on train set: use queries with ≥5 gold citations (most train queries have 4.1 mean) to estimate appropriate K range
- Use Mistral-7B to estimate per-query citation count from query text — this is a query classification task, not a retrieval task
- Report confidence intervals on val Macro-F1 (bootstrapped 10 samples → ±0.05 interval)
- Do not make aggressive hyperparameter changes based solely on val score movements < 0.03

**Phase:** Calibration and LLM augmentation phases

---

### CP-5: rank-bm25 Infeasible for 2.47M Court Corpus

**What goes wrong:** rank-bm25 builds the full TF-IDF matrix in RAM. For 2.47M documents with average ~200 tokens each, the inverted index alone exceeds 8-16GB RAM. Indexing time on Kaggle is 2-4 hours. This consumes almost the entire 12-hour runtime budget before any retrieval happens.

**Research finding:** BM25S (a 2024 library) uses scipy sparse matrices and memory-mapped arrays, achieving 100-500x speedup over rank-bm25 for large corpora. It supports the same BM25 variants (BM25Okapi, BM25+, BM25L) and produces identical results.

**Consequences:** Kaggle runtime exhausted before inference, or OOM crash during BM25 indexing.

**Warning signs:**
- BM25 index construction taking >20 minutes for the 175K laws corpus (should take <2 min with BM25S)
- RAM usage climbing continuously during corpus load
- Notebook kills with OOM during BM25 corpus tokenization

**Prevention:**
- Replace `rank_bm25.BM25Okapi` with `bm25s.BM25` (pip install bm25s)
- For 2.47M court corpus: use `bm25s` with memory-mapped index saved to disk
- Index the court corpus in batches of 500K documents, saving partial indices
- Filter court corpus to German-language documents before indexing (reduces to ~60-70% of 2.47M)
- Cache the built BM25 index to disk and reload on subsequent runs if within same session

**Phase:** Court corpus BM25 phase — critical path for runtime budget

---

### CP-6: Swiss Legal Citation Format Parsing Failures

**What goes wrong:** Swiss legal citations have multiple valid formats and the competition uses exact string matching. Parsing failures produce citations that are semantically correct but string-different from the corpus canonical form.

**Formats in scope:**
- Laws: `Art. 41 OR` / `Art. 41 Abs. 1 OR` / `Art. 41 Abs. 1 lit. a OR`
- BGE (published leading cases): `BGE 144 III 462` (volume, division, page)
- BGer (unpublished docket): `BGer 4A_100/2024` (chamber/docket/year)
- Cantonal references: `VGer ZH VB.2023.123` (court/canton/number/year)

**Edge cases that cause false negatives:**
- `Art. 41 OR` vs `OR Art. 41` (word order variants)
- `Art. 41 Abs.1 OR` vs `Art. 41 Abs. 1 OR` (spacing around abbreviations)
- BGE volume-division-page: "BGE 144 III 462" vs "BGE 144 III, 462" (comma variant)
- Abbreviated law codes: "ZGB" vs "CC" (German vs French abbreviation for Civil Code)
- Roman numerals in BGE division: "BGE 144 III" vs "BGE 144 3" (numeral vs digit)
- Swiss German `ss` vs `ß`: "Strassenverkehrsgesetz" vs "Straßenverkehrsgesetz"

**Consequences:** LLM-generated or slightly reformatted citations fail exact matching, producing FN for semantically correct predictions.

**Warning signs:**
- Manual review shows predicted citations are "almost right" — differ only in spacing or abbreviation
- BGE/BGer citations from Mistral-7B don't match corpus canonical form
- Gold citations in val set have inconsistent formatting (check with `df.citations.str.split(';').explode().str.strip().value_counts()`)

**Prevention:**
- Build a canonicalization function: strip extra whitespace, normalize Roman numerals, standardize `ß→ss`, normalize `Abs.1→Abs. 1`
- After any LLM-generated citation, look it up in corpus → if not found, try normalization variants
- Build a citation fuzzy matcher: Levenshtein distance ≤2 between prediction and corpus citation → snap to corpus form
- Maintain a law code alias table: ZGB↔CC, OR↔CO, SchKG↔LP, StGB↔CP, etc.

**Phase:** Citation parsing and post-processing phase

---

### CP-7: RRF Equal Weighting When Signal Quality Is Unequal

**What goes wrong:** Standard RRF weights all signals equally (BM25 laws, dense laws, BM25 court). In this pipeline, dense retrieval on laws is a strong signal, BM25 on laws is medium, and BM25 on courts (with 2.47M noisy documents) is a weaker, higher-recall signal. Equal weighting causes the noisy court BM25 results to dilute the high-precision dense results.

**Research finding:** RRF is fundamentally a consensus algorithm — it works best when different signals have overlapping results. When signals are disjoint (different corpora), RRF simply interleaves lists rather than fusing them, and the smoothing constant k (default 60) governs which source dominates. Lower k values give massive advantages to top-ranked items from any list; higher k values flatten distinctions. Tuning k is sensitive: several-point NDCG swings occur across k values.

**Consequences:** Precision drops because low-quality court BM25 candidates are ranked alongside high-quality law candidates. Cross-encoder reranker sees more noise in its input, reducing final quality.

**Warning signs:**
- Adding court BM25 signal *decreases* overall val Macro-F1 (net negative)
- Top-10 predicted citations contain unrelated court decisions alongside correct laws
- Cross-encoder reranking takes longer as candidate pool grows with noisy court results

**Prevention:**
- Use weighted RRF: assign weight 1.0 to dense laws, 0.8 to BM25 laws, 0.5 to BM25 courts
- Tune k constant per-signal: use k=30 for dense (reward top ranks), k=60 for BM25 (smooth out noise)
- Keep court BM25 candidate count lower (top-50) vs laws candidates (top-200) before fusion
- Evaluate each signal independently on val before combining — only add signals that improve F1

**Phase:** Fusion and RRF tuning phase

---

### CP-8: Cross-Encoder Reranking Latency Kills Runtime Budget

**What goes wrong:** mMiniLM cross-encoder scores each (query, document) pair independently. At 50ms per pair on GPU, reranking 500 candidates × 50 queries = 25,000 pairs = 1,250 seconds (~21 minutes). If BM25 court retrieval surfaces 2,000 candidates per query and candidate count is not capped, reranking alone exceeds the 12-hour budget.

**Research finding:** Cross-encoder latency scales linearly with candidate set size × sequence length. The mMiniLM model processes 512-token inputs — Swiss legal text with full article + query can easily hit this limit, causing truncation of relevant content.

**Warning signs:**
- Reranking phase takes >30 minutes for 50 queries
- CPU utilization spikes during reranking (model not on GPU)
- Encoded sequences hitting 512-token truncation (visible from tokenizer warning)

**Prevention:**
- Cap cross-encoder input: top-200 candidates from RRF, not top-500
- Batch reranking: score 32 pairs per forward pass, not one-by-one
- Truncate document text to 256 tokens for reranking (keep citation identifier + first paragraph)
- Optionally: skip cross-encoder if BM25 + dense already agree on top-K (high-confidence queries)
- Profile: time the reranking step separately on 10 queries before committing to full run

**Phase:** Reranking implementation phase

---

## Moderate Pitfalls

### MP-1: FAISS Index Type Choice for 175K Laws Corpus

**What goes wrong:** Using IndexFlatIP (exact search, brute force) is correct for 175K docs — it takes ~700MB VRAM and <1s per query. But if the team tries to extend this to 2.47M court docs with the same index type, memory requirement jumps to 10GB VRAM just for the index, causing OOM before any queries run.

**Prevention:**
- Laws corpus (175K): IndexFlatIP is appropriate
- Court corpus (2.47M): do NOT build dense index for full corpus — use BM25 to narrow to top-5K candidates, then encode only those 5K documents for dense reranking
- If FAISS index must be persisted across notebook cells: use `faiss.write_index()` to disk, reload as needed

---

### MP-2: Kaggle Notebook Reproducibility — FAISS HNSW Randomness

**What goes wrong:** FAISS HNSW index construction is non-deterministic by default. Two runs with identical data can produce slightly different recall@K because graph construction uses randomized neighbor selection. This affects reproducibility of Kaggle notebook submissions.

**Prevention:**
- Use IndexFlatIP (deterministic, exact) for the laws corpus where it fits
- If using HNSW: set `index.hnsw.efSearch` and `index.hnsw.efConstruction` explicitly, and set numpy random seed before construction
- For reproducibility: always set `np.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)` at notebook start
- For Kaggle code competitions: non-determinism can cause submission rejection if results differ between runs

---

### MP-3: OpusMT Download Failing in Offline Kaggle Kernel

**What goes wrong:** Kaggle offline notebooks have no internet access. OpusMT and all other models must be downloaded to Kaggle's `/kaggle/input/` from a pre-uploaded dataset or be available via Kaggle's Hugging Face mirror. If the model name is wrong or the HF mirror is outdated, model download fails silently or raises a cryptic error.

**Prevention:**
- Pre-test all model downloads in a separate Kaggle notebook with internet ON
- Add `local_files_only=True` to all `from_pretrained()` calls in offline mode
- Cache models to `/kaggle/working/models/` and check if they exist before downloading
- Verify exact model IDs: `Helsinki-NLP/opus-mt-en-de` (not `opus-mt-tc-big-en-de` — different model)

---

### MP-4: Empty RRF Fusion on Sparse Queries

**What goes wrong:** Queries that are very short or highly specialized (e.g., "What is the penalty for X under ZGB?") may return zero results from one or more retrieval signals. `rrf_fuse()` with an empty input list either crashes (KeyError/IndexError) or skips that signal silently, changing the effective weight of remaining signals without warning.

**Prevention:**
- Add explicit guard: if any ranking list is empty, skip it from fusion (do not pass empty list)
- Log a warning when a retrieval signal returns 0 results: `logger.warning(f"BM25 laws returned 0 results for query {qid}")`
- Implement mandatory fallback: if ALL signals return 0 results, predict the top-10 most frequent citations from training data

---

### MP-5: Submission CSV Format Rejection

**What goes wrong:** The Kaggle evaluator requires exact format: `id,citations` where citations is a semicolon-separated string with no trailing semicolons, no spaces around semicolons, all query IDs present, no NaN values. Any deviation causes immediate rejection with no error message, wasting the 12-hour run.

**Prevention:**
- Add a `validate_submission(df)` function that checks: all test IDs present, no NaN, no empty string citations, semicolons but no commas in citation strings, no duplicate query IDs
- Run validation before `df.to_csv()` and raise ValueError on failure
- Test the validation function on a known-good submission format before any real run
- Cross-check format against the competition's sample_submission.csv exactly

---

### MP-6: Multilingual-E5 Query Prefix Requirements

**What goes wrong:** multilingual-e5-large requires query text to be prefixed with `"query: "` and document text to be prefixed with `"passage: "`. Omitting these prefixes causes the model to treat both query and document identically, severely degrading cosine similarity scores and destroying retrieval quality. This is a documented requirement in the model card that is easy to forget when integrating into a pipeline.

**Warning signs:**
- Dense retrieval quality is worse than BM25 despite using a larger model
- Query and document embeddings cluster together instead of separating by relevance
- All cosine similarities are >0.95 (embeddings not properly separated)

**Prevention:**
- Wrap all encode calls: `encode(["query: " + q for q in queries])` and `encode(["passage: " + d for d in docs])`
- Add unit test: `assert cosine_sim(encode(["query: foo"]), encode(["passage: foo"])) < cosine_sim(encode(["query: foo"]), encode(["passage: foo bar relevant"]))` (approximately)

---

## Minor Pitfalls

### MN-1: French/Italian Court Decisions Under-Retrieved

**What goes wrong:** ~30% of the 2.47M court_considerations.csv is French or Italian text. German-only BM25 tokenization and German stopword removal degrades recall for these documents. Since the val/test queries are English, the cross-lingual gap for FR/IT documents is even larger.

**Mitigation:** Scope to German-only court decisions first (tag by language using `langdetect`). Only add FR/IT handling if German-only pipeline leaves obvious gaps.

---

### MN-2: Citation Deduplication Across Corpora

**What goes wrong:** The same legal article may appear in both laws_de.csv and court_considerations.csv (quoted in a court opinion). Predicting it twice causes duplicate semicolons in submission. While Kaggle scoring may treat duplicates as one, the submission format validator may reject it.

**Mitigation:** Build a global `seen_citations` set during candidate merging; skip adding a citation already in the set regardless of which corpus it came from.

---

### MN-3: Leaderboard Shake-Up Risk

**What goes wrong:** The competition uses 50% public / 50% private leaderboard split. The 40 test queries are split: 20 visible, 20 hidden. If the team over-tunes hyperparameters to the public 20 queries (especially K calibration), the private 20 may have a different citation-count distribution, causing rank drop.

**Mitigation:** "Trust CV" — validate on the 10 val queries, but weight decisions toward robustness over val maximization. Prefer simpler, more robust approaches (entity-driven retrieval, LLM count estimation) over aggressive threshold tuning.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| GPU setup | Silent CPU fallback (AH-1) | Hard assert on cuda.is_available() as first cell |
| BM25 court corpus | rank-bm25 OOM at 2.47M docs (CP-5) | Use BM25S, filter to German-only first |
| BM25 tokenization | German compound blindness (CP-1) | Decompounding + Swiss-specific normalization |
| Translation | OpusMT legal term failures (CP-2) | Dual retrieval: English + German query |
| Dense encoding | Wrong model prefix (MP-6) | Prefix assertions in encode wrapper |
| Dense encoding | e5-small by mistake (AH-4) | Assert embedding dim == 1024 |
| Model loading | VRAM OOM with all models loaded (CP-3) | Sequential load/unload lifecycle |
| Mistral-7B | bfloat16 on T4 crashes (CP-3) | Force float16, eager attention |
| RRF fusion | Equal weights hurting precision (CP-7) | Weighted RRF, evaluate each signal independently |
| Reranking | Cross-encoder latency >budget (CP-8) | Cap at 200 candidates, batch forward passes |
| Swiss citations | Format mismatch exact matching (CP-6) | Canonicalization + fuzzy snap to corpus |
| Calibration | Overfitting K on 10 val queries (CP-4) | Per-query count prediction, not global K |
| Submission | CSV format rejection (MP-5) | validate_submission() before every write |
| Reproducibility | FAISS HNSW non-determinism (MP-2) | Use IndexFlatIP for laws corpus |

---

## Sources

- GerDaLIR (German Legal Information Retrieval dataset): https://github.com/lavis-nlp/GerDaLIR
- BM25S library (faster than rank-bm25): https://huggingface.co/blog/xhluca/bm25s
- Multilingual-E5-Large model card: https://huggingface.co/intfloat/multilingual-e5-large
- bitsandbytes T4 compute capability issues: https://github.com/TimDettmers/bitsandbytes/issues/529
- FAISS guidelines for index selection: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- Weighted RRF in Elasticsearch: https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf
- Cross-encoder latency analysis: https://mbrenndoerfer.com/writing/reranking-cross-encoders-information-retrieval
- Macro-F1 pitfalls with imbalanced data: https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/
- MT hallucination research: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00615/118716
- Stemming impact on German legal retrieval (no-stemming preferred with BM25): https://aclanthology.org/2021.stil-1.25.pdf
- BGE citation format reference: https://dociq.io/glossary/bge
- Cross-lingual retrieval failures in multilingual RAG: https://optyxstack.com/rag-reliability/multilingual-rag-retrieval-fixing-cross-language-misses-without-maintaining-separate-indexes
- OPUS-MT and domain shift: https://link.springer.com/article/10.1007/s10579-023-09704-w
- Kaggle shake-up survival guide: https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-tips-tricks-to-survive-a-kaggle-shake-up-23675beed05e

---

*Confidence note: AH-1 through AH-5 are HIGH confidence (team hit them). CP-1 through CP-8 are MEDIUM-HIGH confidence (verified from primary sources and GerDaLIR/FAISS/bitsandbytes official documentation). MP-* are MEDIUM confidence (research-supported, not team-validated).*

*Last updated: 2026-04-10*
