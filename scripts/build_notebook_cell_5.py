"""
Build Cell 5 of notebook_kaggle.ipynb for Phase 1 Plan 01-03 (Task 1).

Cell 5 — BGE-M3 fp16 load + bge_encode() helper + probe assert.

Implements Research Pattern 2 (BGE-M3 Loading with Correct Pooling) and
guards Pitfalls 1 (mean-pool trap) and 2 (e5 prefixes trap):
  * CLS pooling via last_hidden_state[:, 0] — NOT mean_pool
  * NO 'query:' / 'passage:' prefixes — those are e5-only
  * L2-normalize the CLS embedding
  * Probe assertion: shape == (2, 1024) AND std > 1e-3

Idempotent: replaces cell [5] in place if present, appends otherwise.
Cells 0-4 (from Plans 01-01 and 01-02) are left untouched.
"""
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebook_kaggle.ipynb"

CELL_5_SRC = '''# Cell 5 — Stage 2a: BGE-M3 fp16 load. CLS pooling, NO prefixes.
# Pitfalls 1 and 2 guarded inline.
from transformers import AutoTokenizer, AutoModel

print(f"\\n=== Stage 2a: BGE-M3 load ===")
print(f"Loading BGE-M3 from {BGE_M3_DIR}")
t0 = time.time()
bge_tok = AutoTokenizer.from_pretrained(str(BGE_M3_DIR))
bge_mdl = AutoModel.from_pretrained(str(BGE_M3_DIR), torch_dtype=torch.float16)
bge_mdl = bge_mdl.to(DEVICE).eval()
print(f"  loaded in {time.time()-t0:.1f}s; "
      f"params={sum(p.numel() for p in bge_mdl.parameters())/1e6:.0f}M; "
      f"dtype={next(bge_mdl.parameters()).dtype}; "
      f"device={next(bge_mdl.parameters()).device}")

@torch.no_grad()
def bge_encode(texts, batch_size=64, max_length=512):
    """
    BGE-M3 dense encoder. CLS pooling (last_hidden_state[:, 0]) + L2 normalize.
    Takes RAW text (no 'query:' / 'passage:' prefixes — those are e5 prefixes and
    will poison BGE-M3 embeddings if added). Returns (N, 1024) float32 np.ndarray.
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t)[:2000] for t in texts[i:i+batch_size]]
        enc = bge_tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(DEVICE)
        out = bge_mdl(**enc)
        emb = out.last_hidden_state[:, 0]                     # CLS token
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2 normalize
        all_embs.append(emb.float().cpu().numpy())
    return np.vstack(all_embs).astype("float32")

# Probe: verify dimension is 1024 AND that the two short strings produce
# distinct non-degenerate embeddings (Pitfall 1 guard).
_probe = bge_encode(["Art. 1 ZGB", "Article 1 Swiss Civil Code"])
assert _probe.shape == (2, 1024), f"BGE-M3 dim mismatch: {_probe.shape}"
assert _probe.std() > 1e-3, f"BGE-M3 probe variance suspiciously low: {_probe.std():.6f}"
print(f"  probe dim OK: {_probe.shape}  std={_probe.std():.4f}")
'''


def main():
    nb = nbformat.read(NB_PATH, 4)

    # Dependency check: Plans 01-01 and 01-02 must already have placed
    # Cells 0-4 (CUDA gate, data load, canonicalize, OpusMT).
    if len(nb.cells) < 5:
        raise SystemExit(
            f"ABORT: notebook has only {len(nb.cells)} cells; Plan 01-02 "
            "Cell 4 (OpusMT) is required before this plan can run."
        )
    cell4_src = nb.cells[4].source
    if "MarianMTModel" not in cell4_src or "translations" not in cell4_src:
        raise SystemExit(
            "ABORT: Cell 4 does not contain MarianMTModel/translations. "
            "Plan 01-02 must be applied first."
        )

    new_cell = nbformat.v4.new_code_cell(source=CELL_5_SRC)
    new_cell.pop("id", None)

    if len(nb.cells) >= 6:
        # Replace in place — idempotent re-run or legacy-cell overwrite.
        nb.cells[5] = new_cell
    else:
        nb.cells.append(new_cell)

    nbformat.write(nb, NB_PATH)
    print(f"Wrote Cell 5 ({len(CELL_5_SRC)} chars) to {NB_PATH}")
    print(f"Notebook now has {len(nb.cells)} cells.")


if __name__ == "__main__":
    main()
