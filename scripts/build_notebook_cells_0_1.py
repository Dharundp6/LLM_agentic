"""
Build Cells 0 and 1 of notebook_kaggle.ipynb for Phase 1 Plan 01-01.

Cell 0 — Kaggle pip installs, IS_KAGGLE path switch (incl. BGE_M3_DIR + OPUS_MT_DIR),
feature flags (INC-02), hyperparameter constants, SEED.

Cell 1 — Hard-assert torch.cuda.is_available() (AH-1 / FOUND-01), log device
name and VRAM total/free, deterministic seeds, unload() helper for D-06 model
lifecycle.

This script is idempotent: it opens the existing notebook and REPLACES cells
0 and 1 (or inserts them if missing). All other cells are preserved in place.
"""
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebook_kaggle.ipynb"

CELL_0_SRC = '''# Cell 0 — Kaggle pip installs, path constants, feature flags.
# Runs top-to-bottom. Non-Kaggle (local Windows) path short-circuits installs.

import os, sys, subprocess
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    # Pinned, fail-loud installs. bm25s + CharSplit + nltk are small; faiss-gpu is best-effort.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "bm25s>=0.2.0", "CharSplit", "nltk"])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "faiss-gpu-cu12>=1.14"])
        USE_FAISS_GPU = True
    except subprocess.CalledProcessError:
        print("faiss-gpu-cu12 install failed; falling back to faiss-cpu", flush=True)
        USE_FAISS_GPU = False
    import nltk
    nltk.download("stopwords", quiet=True)
else:
    USE_FAISS_GPU = False  # local Windows dev always CPU faiss

# === Path constants (extends solution.py:20-35 with BGE_M3_DIR + OPUS_MT_DIR) ===
if IS_KAGGLE:
    DATA_DIR    = Path("/kaggle/input/llm-agentic-legal-information-retrieval")
    BGE_M3_DIR  = Path("/kaggle/input/bge-m3")
    OPUS_MT_DIR = Path("/kaggle/input/opus-mt-tc-big-en-de")
    RERANK_DIR  = Path("/kaggle/input/mmarco-reranker")
    OUT_PATH    = Path("/kaggle/working/submission.csv")
else:
    _ROOT = Path(r"C:\\Users\\Dharun prasanth\\OneDrive\\Documents\\Projects\\LLm_Agentic")
    DATA_DIR    = _ROOT / "Data"
    BGE_M3_DIR  = _ROOT / "models" / "bge-m3"
    OPUS_MT_DIR = _ROOT / "models" / "opus-mt-tc-big-en-de"
    RERANK_DIR  = _ROOT / "models" / "mmarco-reranker"
    OUT_PATH    = _ROOT / "submission.csv"

# === Feature flags (INC-02) ===
USE_RERANKER           = True    # Phase 1: on
USE_COURT_CORPUS       = False   # Phase 2+ enables
USE_ENTITY_GRAPH       = False   # Phase 3+ enables
USE_COURT_DENSE_RERANK = False   # Phase 4+ enables
USE_JINA_CROSS_ENCODER = False   # Phase 4+ replaces mmarco
USE_LLM_AUGMENTATION   = False   # Phase 5+ enables
SMOKE                  = os.environ.get("SMOKE", "0") == "1"

# === Hyperparameters (tune on val only) ===
BM25_LAWS_K   = 100
DENSE_LAWS_K  = 100
RRF_K_CONST   = 60
RERANK_K      = 150
SMOKE_LAWS_N  = 5000
SMOKE_VAL_N   = 3
SEED          = 42

print(f"IS_KAGGLE={IS_KAGGLE}  USE_FAISS_GPU={USE_FAISS_GPU}  SMOKE={SMOKE}")
'''

CELL_1_SRC = '''# Cell 1 — HARD GATE: abort if GPU is not present. AH-1 prevention.
import torch, time, gc, re, random
import numpy as np
import pandas as pd

assert torch.cuda.is_available(), (
    "GPU unavailable. Check Kaggle Settings -> Accelerator -> GPU T4 x2. "
    "AH-1 prevention: this notebook refuses to run on CPU."
)

DEVICE = torch.device("cuda")
_dev_name = torch.cuda.get_device_name(0)
_vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
_vram_free  = torch.cuda.mem_get_info()[0] / 1e9
print(f"Device: {_dev_name}")
print(f"VRAM total: {_vram_total:.1f} GB")
print(f"VRAM free:  {_vram_free:.1f} GB")

# Deterministic seeds
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Small helper reused by every model stage (D-06)
def unload(*objs):
    """Free GPU memory: del refs, gc collect, empty_cache, log VRAM free."""
    for _o in objs:
        del _o
    gc.collect()
    torch.cuda.empty_cache()
    _free, _total = torch.cuda.mem_get_info()
    print(f"  unloaded -> VRAM free: {_free/1e9:.1f}/{_total/1e9:.1f} GB", flush=True)
'''


def main():
    nb = nbformat.read(str(NB_PATH), as_version=4)

    cell0 = nbformat.v4.new_code_cell(source=CELL_0_SRC)
    cell1 = nbformat.v4.new_code_cell(source=CELL_1_SRC)

    # Ensure at least 2 cell slots exist; replace the first two.
    if len(nb.cells) == 0:
        nb.cells = [cell0, cell1]
    elif len(nb.cells) == 1:
        nb.cells[0] = cell0
        nb.cells.insert(1, cell1)
    else:
        nb.cells[0] = cell0
        nb.cells[1] = cell1

    nbformat.write(nb, str(NB_PATH))
    print(f"Wrote Cells 0 and 1 to {NB_PATH}")
    print(f"Total cells now: {len(nb.cells)}")


if __name__ == "__main__":
    main()
