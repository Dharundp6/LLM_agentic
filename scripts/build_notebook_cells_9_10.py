"""
Idempotent nbformat editor that installs Cells 9 and 10 in notebook_kaggle.ipynb.

Cell 9  = Stage 4: RRF fuse dense_laws_ids + bm25_laws_ids per query.
Cell 10 = Stage 5: mmarco cross-encoder rerank top-RERANK_K fused candidates.

Plan: 01-foundation-laws-pipeline / 01-05 Task 1
Depends on Plans 01-01..01-04 having already installed Cells 0-8.

Running this script is safe on either starting state:
  * if notebook has exactly 9 cells (0-8): Cells 9 and 10 are appended.
  * if notebook already has >=11 cells: Cells 9 and 10 are replaced in place.
"""
from __future__ import annotations

from pathlib import Path
import sys

import nbformat


NOTEBOOK = Path("notebook_kaggle.ipynb")
CELL9_INDEX = 9
CELL10_INDEX = 10

CELL9_SOURCE = '''# Cell 9 — Stage 4: RRF fuse dense_laws_ids + bm25_laws_ids per query.
# Reciprocal Rank Fusion combines the two laws recall signals into one ranked
# candidate list per query. RRF_K_CONST=60 matches solution.py:220-226.

def rrf_fuse(rankings, k=60):
    """Reciprocal rank fusion. Returns list[(doc_id, score)] sorted desc."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

print(f"\\n=== Stage 4: RRF fuse (laws dense + laws bm25) ===")
t0 = time.time()
fused_laws_ids = {}  # qid -> list[int] of laws row indices, sorted by fused score
for qid in _target_ids:
    fused = rrf_fuse([dense_laws_ids[qid], bm25_laws_ids[qid]], k=RRF_K_CONST)
    fused_laws_ids[qid] = [int(doc_id) for doc_id, _score in fused]
print(f"Fused {len(fused_laws_ids)} queries  (dense+bm25, RRF k={RRF_K_CONST})  [{time.time()-t0:.1f}s]")
print(f"  sample qid={_target_ids[0]}  fused top-5={fused_laws_ids[_target_ids[0]][:5]}")
'''

CELL10_SOURCE = '''# Cell 10 — Stage 5: mmarco cross-encoder rerank top-RERANK_K fused candidates.
# Loads mmarco-mMiniLMv2 AFTER BGE-M3 is fully unloaded (D-06, enforced by Cell 7).
# USE_RERANKER gates the whole stage (INC-02).

from transformers import AutoModelForSequenceClassification

print(f"\\n=== Stage 5: Cross-encoder rerank ===")

if USE_RERANKER:
    t0 = time.time()
    rerank_tok = AutoTokenizer.from_pretrained(str(RERANK_DIR))
    rerank_mdl = AutoModelForSequenceClassification.from_pretrained(str(RERANK_DIR))
    rerank_mdl = rerank_mdl.to(DEVICE).eval()
    print(f"Reranker loaded in {time.time()-t0:.1f}s; "
          f"device={next(rerank_mdl.parameters()).device}")

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

    # Per query: take top-RERANK_K fused candidates, score with cross-encoder, re-sort.
    reranked_laws_ids = {}
    t0 = time.time()
    for qid in _target_ids:
        cand_ids = fused_laws_ids[qid][:RERANK_K]
        if not cand_ids:
            reranked_laws_ids[qid] = []
            continue
        cand_texts = [
            f"{laws.iloc[idx]['citation']} {str(laws.iloc[idx]['text'])[:1500]}"
            for idx in cand_ids
        ]
        scores = rerank(q_en_canon[qid], cand_texts, batch_size=32)
        order = np.argsort(-scores)
        reranked_laws_ids[qid] = [int(cand_ids[i]) for i in order]
    print(f"Reranked {len(reranked_laws_ids)} queries in {time.time()-t0:.1f}s")

    unload(rerank_tok, rerank_mdl)
else:
    print("USE_RERANKER=False; skipping rerank, using fused order as-is.")
    reranked_laws_ids = {qid: fused_laws_ids[qid][:RERANK_K] for qid in _target_ids}
'''


def _install(nb, index: int, source: str) -> None:
    cell = nbformat.v4.new_code_cell(source=source)
    # nbformat_minor=4 on this notebook rejects the 'id' attribute that
    # nbformat.v4.new_code_cell sets by default; pop it so nbformat.read
    # stays clean (matches Plan 01-02 / 01-03 / 01-04 convention).
    cell.pop("id", None)
    if len(nb.cells) >= index + 1:
        print(f"Replacing existing Cell {index} in place")
        nb.cells[index] = cell
    else:
        print(f"Appending new Cell {index}")
        nb.cells.append(cell)


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"ERROR: {NOTEBOOK} not found (run from worktree root)", file=sys.stderr)
        return 1

    nb = nbformat.read(NOTEBOOK, as_version=4)

    # Dependency check: Plans 01-01..01-04 must have installed Cells 0-8.
    if len(nb.cells) < 9:
        print(
            f"ERROR: notebook has only {len(nb.cells)} cells; expected >=9 from Plans 01-01..01-04",
            file=sys.stderr,
        )
        return 2
    if "dense_laws_ids" not in nb.cells[7].source:
        print(
            "ERROR: Cell 7 missing dense_laws_ids — Plan 01-03 not applied",
            file=sys.stderr,
        )
        return 3
    if "bm25_laws_ids" not in nb.cells[8].source:
        print(
            "ERROR: Cell 8 missing bm25_laws_ids — Plan 01-04 not applied",
            file=sys.stderr,
        )
        return 4
    if "_target_ids" not in nb.cells[7].source:
        print(
            "ERROR: Cell 7 missing _target_ids — Plan 01-03 not applied",
            file=sys.stderr,
        )
        return 5

    _install(nb, CELL9_INDEX, CELL9_SOURCE)
    _install(nb, CELL10_INDEX, CELL10_SOURCE)

    nbformat.write(nb, NOTEBOOK)
    print(f"OK: notebook now has {len(nb.cells)} cells; Cells 9 and 10 installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
