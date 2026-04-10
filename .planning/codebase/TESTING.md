# Testing Patterns

**Analysis Date:** 2026-04-10

## Test Framework

**Runner:**
- No formal test framework (pytest, unittest, vitest not found)
- Custom evaluation scripts instead
- Jupyter notebook (`notebook_kaggle.ipynb`) serves as both development and validation environment

**Validation Approach:**
- Validation set (`Data/val.csv`) used for offline evaluation
- Held-out test set (`Data/test.csv`) for final submission
- Cross-validation not used; single train/val/test split

**Assertion Library:**
- Standard Python `assert` for data validation: `assert DATA_DIR is not None, 'laws_de.csv not found'`
- No formal assertion library

**Run Commands:**
```bash
# BM25-only baseline evaluation (fast, no GPU)
python eval_local.py              # TF-IDF baseline val macro-F1

# Dense retrieval test on multilingual-e5-small (CPU)
python eval_dense.py              # E5-small cross-lingual validation

# Full production pipeline (Kaggle with GPU)
# Run notebook_kaggle.ipynb in Kaggle environment

# Data diagnosis and analysis
python diagnose.py                # Gold citation corpus coverage
python diagnose3.py               # Train/val citation overlap
python diagnose6.py               # [diagnostic output]

# Full solution pipeline (local or Kaggle)
python solution.py                # Complete retrieval + reranking pipeline
```

## Test File Organization

**Location:**
- Evaluation scripts in project root: `/eval_*.py`
- Diagnostic scripts in project root: `/diagnose*.py`
- Co-located with source: No separate `tests/` directory
- Jupyter development: `notebook_kaggle.ipynb` embeds validation steps

**Naming:**
- Evaluation scripts: `eval_*.py` (e.g., `eval_local.py`, `eval_dense.py`)
- Diagnostic scripts: `diagnose*.py` with numeric suffix (e.g., `diagnose.py`, `diagnose3.py`, `diagnose6.py`)
- Pattern indicates iterative debugging/analysis

**Structure:**
```
LLm_Agentic/
├── solution.py              # Main production pipeline
├── eval_local.py            # BM25 baseline validation
├── eval_dense.py            # Dense (E5) validation
├── diagnose.py              # Gold citation corpus check
├── diagnose2.py             # [variant analysis]
├── diagnose3.py             # Train/val overlap analysis
├── diagnose4.py             # [variant analysis]
├── diagnose5.py             # [variant analysis]
├── diagnose6.py             # [variant analysis]
├── notebook_kaggle.ipynb    # Kaggle notebook (dev + inference)
├── requirements.txt         # Dependencies
└── Data/
    ├── laws_de.csv          # German law corpus
    ├── court_considerations.csv  # Court decisions corpus
    ├── train.csv            # Training queries + gold citations
    ├── val.csv              # Validation queries + gold citations
    └── test.csv             # Test queries (no gold labels)
```

## Test Structure

**Suite Organization:**

From `eval_local.py`:
```python
# Data loading
laws = pd.read_csv(DATA / "laws_de.csv")
court = pd.read_csv(DATA / "court_considerations.csv")
val = pd.read_csv(DATA / "val.csv")

# Setup: Build retrieval indices
vec_laws = TfidfVectorizer(...)
X_laws = vec_laws.fit_transform(laws_texts)

# Execution: Run retrieval on val queries
for i, row in val.iterrows():
    cits = retrieve(row["query"], top_k=60)
    gold = set(row["gold_citations"].split(";"))
    val_results.append({"gold": gold, "cits": cits})

# Evaluation: Compute macro-F1 across different top-K thresholds
for k in range(1, 60):
    f1 = macro_f1([r["gold"] for r in val_results],
                  [set(r["cits"][:k]) for r in val_results])
    if f1 > best_f1:
        best_f1, best_k = f1, k
```

**Patterns:**

1. **Setup Phase:**
   - Load corpora once: `laws = pd.read_csv(...)`
   - Build indices: `TfidfVectorizer.fit_transform()` or `BM25Okapi()`
   - Load models: `AutoTokenizer.from_pretrained()`, `AutoModel.from_pretrained()`

2. **Execution Phase:**
   - Iterate over queries: `for _, row in val.iterrows():`
   - Retrieve candidates: Call retrieval function with English query
   - Collect results: Store gold set and predicted citations in list

3. **Evaluation Phase:**
   - Compute metrics: `macro_f1(gold_sets, pred_sets)` over all queries
   - Optimize threshold: Sweep `top_k` parameter to maximize F1 on validation
   - Report results: Print best K and corresponding F1 score

## Mocking

**Framework:**
- Not used; no external API calls or file I/O mocking
- All test data comes from actual CSV files loaded into DataFrames
- Model loading from HuggingFace hub without stubbing

**Patterns:**
- No mocking observed in codebase
- All components (BM25, dense encoding, reranking) are real implementations tested on actual data

**What to Mock (if tests were added):**
- HuggingFace model downloads (slow, network-dependent)
- GPU device availability (test on CPU with mocked cuda checks)
- Large corpus encoding (use subset for unit tests)

**What NOT to Mock:**
- Core retrieval logic (must test with real BM25 scores and embeddings)
- Evaluation metrics (must use actual gold/predicted sets)
- Data loading (use fixture CSV files instead)

## Fixtures and Factories

**Test Data:**

From `diagnose.py` (corpus validation):
```python
all_cits = set(laws['citation'].astype(str).tolist()) | set(court['citation'].astype(str).tolist())
for _, row in val.iterrows():
    gold = row['gold_citations'].split(';')
    found = [c for c in gold if c in all_cits]
    missing = [c for c in gold if c not in all_cits]
```

From `diagnose3.py` (train frequency baseline):
```python
train_cits = Counter()
for _, row in train.iterrows():
    for c in str(row["gold_citations"]).split(";"):
        c = c.strip()
        if c:
            train_cits[c] += 1

gold_sets = [set(r["gold_citations"].split(";")) for _, r in val.iterrows()]
top_k = [c for c, _ in train_cits.most_common(k)]
pred_sets = [set(top_k) for _ in val.index]
```

**Location:**
- Fixtures in project root: `Data/val.csv`, `Data/test.csv`, `Data/train.csv`
- No factory pattern; direct CSV loading via pandas
- Fixtures are competition data, not generated test data

**Test Citation Sets:**

Legal citation format: `"Art. 221 Abs. 1 StPO"` (Article, Section, Swiss law code)
Example gold citations from validation:
- Semicolon-separated: `"Art. 221 Abs. 1 StPO;Art. 222 StPO;BGE 99 IV 86"`
- Parsed at runtime: `row['gold_citations'].split(';')`

## Coverage

**Requirements:** 
- None enforced; no coverage configuration file found
- Ad-hoc validation on held-out val set (10 queries)

**View Coverage:**
- No automated coverage measurement
- Manual evaluation via macro-F1 metric computed in eval scripts
- Per-query F1 at top-20: `print(f"gold={len(gold)} pred@20={len(pred20)} F1@20={f1:.3f}")`

## Test Types

**Validation Tests (on val set):**

Scope: 10 English queries + gold German legal citations
Approach: Retrieve top-K candidates and measure macro-F1

Example from `eval_local.py`:
```python
for i, row in val.iterrows():
    cits = retrieve(row["query"], top_k=60)
    gold = set(row["gold_citations"].split(";"))
    val_results.append({"gold": gold, "cits": cits})
    pred20 = set(cits[:20])
    tp = len(gold & pred20)
    pr = tp / len(pred20) if pred20 else 0
    rc = tp / len(gold) if gold else 0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
    log(f"  {row['query_id']}: gold={len(gold)} pred@20={len(pred20)} tp={tp} F1@20={f1:.3f}")
```

**Diagnostic Tests (data quality):**

From `diagnose.py`:
- Corpus coverage: Verify all gold citations exist in laws + court corpus
- Missing citations: Flag gold references not in retrieval pool
- Output: `"Total: {total_found}/{total_gold} = {100*total_found/total_gold:.1f}% gold citations exist in corpus"`

**Baseline Tests:**

From `diagnose3.py`:
- Frequency baseline: Predict top-N most common train citations for all queries
- Train/val overlap: Measure what % of val gold citations appeared in training
- Goal: Understand lower bound performance without ML

**Pipeline Tests (end-to-end):**

From `solution.py` and `notebook_kaggle.ipynb`:
- Full retrieval pipeline: BM25 + dense + RRF fusion + reranking
- Validation: Calibrate top-K on validation set to maximize macro-F1
- Test inference: Generate `submission.csv` with predicted citations for test queries

## Common Patterns

**Metric Computation:**

From `eval_local.py`, `diagnose3.py`, `eval_dense.py`:
```python
def macro_f1(gold_sets, pred_sets):
    f1s = []
    for g, p in zip(gold_sets, pred_sets):
        if not g and not p: f1s.append(1.0); continue
        if not g or not p:  f1s.append(0.0); continue
        tp = len(g & p)
        pr = tp / len(p)
        rc = tp / len(g)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))
```

**Threshold Optimization:**

Pattern across all eval scripts:
```python
best_f1, best_k = 0.0, 0
for k in range(1, MAX_K):
    pred_sets = [set(r["ranked_citations"][:k]) for r in results]
    f1 = macro_f1(gold_sets, pred_sets)
    if f1 > best_f1:
        best_f1, best_k = f1, k
print(f"*** {metric_name} val macro-F1 = {best_f1:.4f} @ top-{best_k} ***")
```

**Batch Processing for Performance:**

From `solution.py`:
```python
@torch.no_grad()
def encode(tokenizer, model, texts, batch_size=64, prefix="passage: "):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t[:500] for t in texts[i : i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(DEVICE)
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())
        if i % 50000 == 0 and i > 0:
            print(f"  Dense encoded {i}/{len(texts)}")
    return np.vstack(all_embs).astype("float32")
```

**GPU/Device Management:**

From `eval_dense.py`, `notebook_kaggle.ipynb`:
```python
DEVICE = torch.device("cpu")  # Explicit CPU choice
torch.set_num_threads(4)

# Or conditional:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup:
mdl = AutoModel.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()
```

**Memory Cleanup:**

Pattern in `solution.py` and notebook:
```python
del laws_snippets; gc.collect()
del embs; gc.collect()
```

## Integration Testing Strategy

**Environment Validation:**

`solution.py` detects Kaggle vs. local:
```python
IS_KAGGLE = os.path.exists("/kaggle")
if IS_KAGGLE:
    DATA_DIR = Path("/kaggle/input/llm-agentic-legal-information-retrieval")
    E5_DIR = Path("/kaggle/input/multilingual-e5-large")
else:
    DATA_DIR = Path(r"C:\Users\Dharun prasanth\OneDrive\Documents\Projects\LLm_Agentic\Data")
    E5_DIR = "intfloat/multilingual-e5-large"  # HF hub
```

**Cross-Environment Compatibility:**

`notebook_kaggle.ipynb` uses dynamic path discovery:
```python
def find_dir_by_name(root, sub):
    s = sub.lower()
    for p in Path(root).rglob('*'):
        if p.is_dir() and s in p.name.lower():
            return p
    return None

DATA_DIR = find_file_dir('/kaggle/input', 'laws_de.csv')
E5_DIR   = find_dir_by_name('/kaggle/input', 'e5-small')
assert DATA_DIR is not None, 'laws_de.csv not found'
```

---

*Testing analysis: 2026-04-10*
