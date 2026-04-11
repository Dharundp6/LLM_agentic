"""
Build Cell 6 of notebook_kaggle.ipynb for Phase 1 Plan 01-03 (Task 2).

Cell 6 — Encode laws corpus (LAWS-01/LAWS-02) with the 60-second / 1000-doc
AH-1 checkpoint (Research Pattern 3) that aborts within 60s if a silent CPU
fallback is detected. Builds a FAISS IndexFlatIP (GPU on Kaggle when
USE_FAISS_GPU is True, CPU fallback otherwise).

Idempotent: replaces cell [6] in place if present, appends otherwise.
Cells 0-5 (Plans 01-01, 01-02, and Plan 01-03 Task 1) are left untouched.
"""
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebook_kaggle.ipynb"

CELL_6_SRC = '''# Cell 6 — Stage 2b: Encode laws corpus (LAWS-01/LAWS-02) with AH-1 checkpoint.
import faiss

print(f"\\n=== Stage 2b: Encode laws corpus ===")

# Smoke mode: truncate to SMOKE_LAWS_N docs.
if SMOKE:
    print(f"SMOKE mode: truncating laws corpus to {SMOKE_LAWS_N} docs")
    laws = laws.iloc[:SMOKE_LAWS_N].reset_index(drop=True)

laws_texts_for_dense = [
    f"{row['citation']} {str(row['text'])[:1500]}"
    for _, row in laws.iterrows()
]
print(f"Encoding {len(laws_texts_for_dense):,} laws docs with BGE-M3 fp16...")

t_start = time.time()
checkpoint_fired = False
all_embs = []
BATCH = 64
for i in range(0, len(laws_texts_for_dense), BATCH):
    batch = [str(t)[:2000] for t in laws_texts_for_dense[i:i+BATCH]]
    enc = bge_tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = bge_mdl(**enc)
    emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
    all_embs.append(emb.float().cpu().numpy())

    # AH-1 checkpoint: at 60s wall-clock, require >=1000 docs encoded.
    if not checkpoint_fired and time.time() - t_start > 60:
        encoded_so_far = min(i + BATCH, len(laws_texts_for_dense))
        assert encoded_so_far >= 1000, (
            f"AH-1 TRIGGER: only {encoded_so_far} docs encoded in 60s. "
            f"Silent CPU fallback suspected. Aborting run."
        )
        checkpoint_fired = True
        print(f"  60s checkpoint OK: {encoded_so_far} docs encoded, proceeding", flush=True)

    if (i // BATCH) % 200 == 0 and i > 0:
        elapsed = time.time() - t_start
        rate = (i + BATCH) / elapsed
        eta = (len(laws_texts_for_dense) - i - BATCH) / max(rate, 1e-9)
        print(f"  {i+BATCH:,}/{len(laws_texts_for_dense):,}  "
              f"[{elapsed:.0f}s, {rate:.0f} doc/s, ETA {eta:.0f}s]", flush=True)

laws_embs = np.vstack(all_embs).astype("float32")
print(f"Encoded laws: {laws_embs.shape}  [{time.time()-t_start:.1f}s total]")
del all_embs; gc.collect()

# Build FAISS IndexFlatIP (LAWS-02). 175K x 1024 ~= 717 MB.
assert laws_embs.shape[1] == 1024, f"BGE-M3 dim mismatch: {laws_embs.shape}"
print("Building FAISS IndexFlatIP...")
t0 = time.time()
cpu_index = faiss.IndexFlatIP(1024)
if USE_FAISS_GPU:
    try:
        _faiss_res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(_faiss_res, 0, cpu_index)
        print("  FAISS on GPU")
    except Exception as _e:
        print(f"  faiss GPU build failed ({_e}); using CPU IndexFlatIP")
        faiss_index = cpu_index
else:
    faiss_index = cpu_index
    print("  FAISS on CPU")
faiss_index.add(laws_embs)
print(f"  FAISS: {faiss_index.ntotal:,} vectors, d={faiss_index.d}  [{time.time()-t0:.1f}s]")
'''


def main():
    nb = nbformat.read(NB_PATH, 4)

    # Dependency check: Plan 01-03 Task 1 must already have placed Cell 5.
    if len(nb.cells) < 6:
        raise SystemExit(
            f"ABORT: notebook has only {len(nb.cells)} cells; Plan 01-03 "
            "Task 1 (Cell 5 BGE-M3 load) is required before this task can run."
        )
    cell5_src = nb.cells[5].source
    if "bge_encode" not in cell5_src or "last_hidden_state[:, 0]" not in cell5_src:
        raise SystemExit(
            "ABORT: Cell 5 does not contain bge_encode / CLS pooling. "
            "Plan 01-03 Task 1 must be applied first."
        )

    new_cell = nbformat.v4.new_code_cell(source=CELL_6_SRC)
    new_cell.pop("id", None)

    if len(nb.cells) >= 7:
        nb.cells[6] = new_cell
    else:
        nb.cells.append(new_cell)

    nbformat.write(nb, NB_PATH)
    print(f"Wrote Cell 6 ({len(CELL_6_SRC)} chars) to {NB_PATH}")
    print(f"Notebook now has {len(nb.cells)} cells.")


if __name__ == "__main__":
    main()
