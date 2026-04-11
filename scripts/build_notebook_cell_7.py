"""
Build Cell 7 of notebook_kaggle.ipynb for Phase 1 Plan 01-03 (Task 3).

Cell 7 — Dual-query dense retrieval (D-04 / QUERY-03). Per query, encode
BOTH q_en_canon AND translations[qid], average the two 1024-dim vectors,
L2 re-normalize, and search the FAISS index for top DENSE_LAWS_K. Stores
results in dense_laws_ids dict for Plans 01-04 and 01-05 to consume.

Then D-06: unload BGE-M3 BEFORE any subsequent cell loads another model.
FAISS index lives on CPU/GPU-faiss resources and survives the unload.
A VRAM-free >= 10 GB assert blocks the next stage if the unload leaks.

Idempotent: replaces cell [7] in place if present, appends otherwise.
"""
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebook_kaggle.ipynb"

CELL_7_SRC = '''# Cell 7 — Stage 2c: Dual-query dense retrieval (D-04 / QUERY-03).
# Per query, encode BOTH q_en_canon AND translations[qid], average the two
# 1024-dim vectors, L2 re-normalize, and search FAISS for top DENSE_LAWS_K.

print(f"\\n=== Stage 2c: Dual-query dense retrieval (laws) ===")

# Decide which queries to process (smoke mode shrinks val, but all test is processed)
_val_ids  = val.iloc[:SMOKE_VAL_N]["query_id"].tolist() if SMOKE else val["query_id"].tolist()
_test_ids = test["query_id"].tolist()
_target_ids = _val_ids + _test_ids
print(f"Processing {len(_target_ids)} queries  (val={len(_val_ids)}, test={len(_test_ids)})")

def dense_query_embedding(qid):
    """D-04: Encode EN + DE, average, re-normalize to unit length."""
    en = q_en_canon[qid]
    de = translations[qid]
    pair = bge_encode([en, de])             # (2, 1024)
    avg = pair.mean(axis=0, keepdims=True)  # (1, 1024)
    norm = np.linalg.norm(avg, axis=1, keepdims=True)
    return (avg / np.maximum(norm, 1e-9)).astype("float32")

dense_laws_ids = {}  # qid -> list[int] of top DENSE_LAWS_K laws row indices

t0 = time.time()
for qid in _target_ids:
    q_vec = dense_query_embedding(qid)
    D, I = faiss_index.search(q_vec, DENSE_LAWS_K)
    dense_laws_ids[qid] = I[0].tolist()

print(f"Dense retrieval done in {time.time()-t0:.1f}s  "
      f"({len(_target_ids)} queries x top-{DENSE_LAWS_K})")
print(f"  sample qid={_target_ids[0]}  top-5 laws idx={dense_laws_ids[_target_ids[0]][:5]}")

# D-06: unload BGE-M3 BEFORE cross-encoder stage.
# FAISS index lives on CPU or GPU-faiss resources; it survives the unload.
_free_before = torch.cuda.mem_get_info()[0] / 1e9
print(f"  VRAM free before BGE-M3 unload: {_free_before:.1f} GB")
unload(bge_tok, bge_mdl)
_free_after = torch.cuda.mem_get_info()[0] / 1e9
assert _free_after >= 10.0, (
    f"BGE-M3 unload incomplete: VRAM free={_free_after:.1f} GB (expected >= 10). "
    f"Cross-encoder load will be starved — investigate."
)
print(f"  VRAM free after  BGE-M3 unload: {_free_after:.1f} GB  [OK]")
'''


def main():
    nb = nbformat.read(NB_PATH, 4)

    # Dependency check: Plan 01-03 Tasks 1 & 2 must already have placed Cells 5 & 6.
    if len(nb.cells) < 7:
        raise SystemExit(
            f"ABORT: notebook has only {len(nb.cells)} cells; Plan 01-03 "
            "Tasks 1-2 (Cells 5 and 6) are required before this task can run."
        )
    cell6_src = nb.cells[6].source
    if "faiss_index" not in cell6_src or "laws_embs" not in cell6_src:
        raise SystemExit(
            "ABORT: Cell 6 does not contain faiss_index / laws_embs. "
            "Plan 01-03 Task 2 must be applied first."
        )

    new_cell = nbformat.v4.new_code_cell(source=CELL_7_SRC)
    new_cell.pop("id", None)

    if len(nb.cells) >= 8:
        nb.cells[7] = new_cell
    else:
        nb.cells.append(new_cell)

    nbformat.write(nb, NB_PATH)
    print(f"Wrote Cell 7 ({len(CELL_7_SRC)} chars) to {NB_PATH}")
    print(f"Notebook now has {len(nb.cells)} cells.")


if __name__ == "__main__":
    main()
