"""
Idempotent nbformat editor that installs Cell 8 in notebook_kaggle.ipynb.

Cell 8 = Stage 3: bm25s laws index build + per-query BM25 retrieval.
Consumes tokenize_for_bm25_de (Cell 3), bm25_query_texts (Cell 4), and
_target_ids (Cell 7). Produces bm25_laws (bm25s index) and bm25_laws_ids
(dict qid -> list[int] of top BM25_LAWS_K laws row indices).

Plan: 01-foundation-laws-pipeline / 01-04
Depends on Plans 01-01, 01-02, 01-03 having already installed Cells 0-7.

Running this script is safe on either starting state:
  * if notebook has exactly 8 cells (0-7): Cell 8 is appended.
  * if notebook already has 9+ cells with an existing Cell 8: Cell 8 is
    replaced in place.
"""
from __future__ import annotations

from pathlib import Path
import sys

import nbformat


NOTEBOOK = Path("notebook_kaggle.ipynb")
CELL_INDEX = 8

CELL_SOURCE = '''# Cell 8 — Stage 3: bm25s laws index build + per-query BM25 retrieval.
# LAWS-03 explicit: German decompounding (via CharSplit inside tokenize_for_bm25_de),
# NO Snowball stemming (PITFALLS CP-1). Pre-tokenized list-of-list input.

import bm25s

print(f"\\n=== Stage 3: bm25s laws index ===")

# Build the tokenized corpus. `laws` was already canonicalized in Cell 3 (LAWS-05).
t0 = time.time()
laws_bm25_texts = [
    f"{row['citation']} {row['text']}"
    for _, row in laws.iterrows()
]
laws_bm25_tokens = [tokenize_for_bm25_de(t) for t in laws_bm25_texts]
print(f"  tokenized {len(laws_bm25_tokens):,} laws docs  [{time.time()-t0:.1f}s]")

# Guardrail: confirm 'Art.' survives the tokenizer for a representative doc.
# If this fails, the CharSplit / stopword / regex interaction is broken.
_smoke_sample = tokenize_for_bm25_de("Die Pflicht aus Art. 41 OR")
assert "art." in _smoke_sample, (
    f"tokenize_for_bm25_de dropped 'Art.': {_smoke_sample}. "
    f"LAWS-03 regression — investigate regex/stopword interaction."
)

# Build bm25s index. Pass list-of-list tokens directly; do NOT use bm25s.tokenize().
t0 = time.time()
bm25_laws = bm25s.BM25()
bm25_laws.index(laws_bm25_tokens)
print(f"  indexed                            [{time.time()-t0:.1f}s]")

# Per-query BM25 retrieval. Uses bm25_query_texts[qid] from Cell 4 (D-03).
print(f"Retrieving BM25 top-{BM25_LAWS_K} for {len(_target_ids)} queries...")
t0 = time.time()
bm25_laws_ids = {}
for qid in _target_ids:
    q_text = bm25_query_texts[qid]
    q_tokens = tokenize_for_bm25_de(q_text)
    if not q_tokens:
        bm25_laws_ids[qid] = []
        continue
    results, scores = bm25_laws.retrieve([q_tokens], k=BM25_LAWS_K)
    # bm25s returns numpy int arrays; cast to plain list[int]
    bm25_laws_ids[qid] = [int(x) for x in results[0]]
print(f"BM25 retrieval done in {time.time()-t0:.1f}s")
print(f"  sample qid={_target_ids[0]}  top-5 bm25 laws idx={bm25_laws_ids[_target_ids[0]][:5]}")
'''


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"ERROR: {NOTEBOOK} not found (run from worktree root)", file=sys.stderr)
        return 1

    nb = nbformat.read(NOTEBOOK, as_version=4)

    # Dependency check: Plans 01-01/02/03 must have installed Cells 0-7.
    if len(nb.cells) < 8:
        print(
            f"ERROR: notebook has only {len(nb.cells)} cells; expected >=8 from Plans 01-01..01-03",
            file=sys.stderr,
        )
        return 2
    if "def tokenize_for_bm25_de" not in nb.cells[3].source:
        print(
            "ERROR: Cell 3 missing tokenize_for_bm25_de — Plan 01-01 not applied",
            file=sys.stderr,
        )
        return 3
    if "bm25_query_texts" not in nb.cells[4].source:
        print(
            "ERROR: Cell 4 missing bm25_query_texts — Plan 01-02 not applied",
            file=sys.stderr,
        )
        return 4
    if "_target_ids" not in nb.cells[7].source:
        print(
            "ERROR: Cell 7 missing _target_ids — Plan 01-03 not applied",
            file=sys.stderr,
        )
        return 5

    new_cell = nbformat.v4.new_code_cell(source=CELL_SOURCE)
    # nbformat_minor=4 on this notebook rejects the 'id' attribute that
    # nbformat.v4.new_code_cell sets by default; pop it so nbformat.read
    # stays clean (matches Plan 01-02 and 01-03 convention).
    new_cell.pop("id", None)

    if len(nb.cells) >= CELL_INDEX + 1:
        print(f"Replacing existing Cell {CELL_INDEX} in place")
        nb.cells[CELL_INDEX] = new_cell
    else:
        print(f"Appending new Cell {CELL_INDEX}")
        nb.cells.append(new_cell)

    nbformat.write(nb, NOTEBOOK)
    print(f"OK: notebook now has {len(nb.cells)} cells; Cell {CELL_INDEX} installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
