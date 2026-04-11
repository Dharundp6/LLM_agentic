"""
Build Cell 4 of notebook_kaggle.ipynb for Phase 1 Plan 01-02.

Cell 4 — OpusMT (opus-mt-tc-big-en-de) EN->DE translation of all val+test
queries. Builds three cache dicts:
  * q_en_canon        — canonicalize(english_original)     (dense + reranker)
  * translations      — canonicalize(german_translation)   (dense dual-query)
  * bm25_query_texts  — f"{de_canon} {extract_legal_codes(english_original)}"

Enforces D-01 (OpusMT in Phase 1), D-02/D-03/D-04 (cache dict contract),
D-05 (canonicalize both EN and DE), and D-06 (single-resident GPU — the
model is fully unloaded before Cell 5 / BGE-M3 loads, with a VRAM-free
guard that fails loudly on incomplete unload, see research Pitfall 4).

Idempotent: replaces cell [4] in place. Cells 0-3 (from Plan 01-01) and
any trailing legacy cells (5+) are left untouched — those are Plan 01-03+
scope.
"""
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebook_kaggle.ipynb"

CELL_4_SRC = '''# Cell 4 — Stage 1: OpusMT EN -> DE translation of all val+test queries.
# D-01/D-02/D-03/D-04/D-05/D-06 enforced inline.

from transformers import AutoTokenizer, MarianMTModel

print(f"\\n=== Stage 1: OpusMT translation ===")
print(f"Loading OpusMT from {OPUS_MT_DIR}")
t0 = time.time()
opus_tok = AutoTokenizer.from_pretrained(str(OPUS_MT_DIR))
opus_mdl = MarianMTModel.from_pretrained(str(OPUS_MT_DIR)).to(DEVICE).eval()
print(f"  loaded in {time.time()-t0:.1f}s; "
      f"device={next(opus_mdl.parameters()).device}; "
      f"dtype={next(opus_mdl.parameters()).dtype}; "
      f"params={sum(p.numel() for p in opus_mdl.parameters())/1e6:.0f}M")

@torch.no_grad()
def translate_en_de(texts, batch_size=8, max_len=512, num_beams=4):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = opus_tok(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len).to(DEVICE)
        gen = opus_mdl.generate(**enc, max_new_tokens=max_len, num_beams=num_beams)
        out.extend([opus_tok.decode(g, skip_special_tokens=True) for g in gen])
    return out

# Translate all val + test queries in one go (~50 total queries)
all_query_ids = val["query_id"].tolist() + test["query_id"].tolist()
all_queries_en_raw = val["query"].tolist() + test["query"].tolist()

t0 = time.time()
all_queries_de_raw = translate_en_de(all_queries_en_raw, batch_size=8, num_beams=4)
print(f"Translated {len(all_queries_en_raw)} queries in {time.time()-t0:.1f}s")

# D-05: canonicalize BOTH English original and German translation before downstream use.
# D-03: append regex-extracted legal codes (from the ORIGINAL English query — codes
# are language-agnostic) to the German translation for BM25 input.
q_en_canon        = {}  # qid -> canonicalize(english)           — used by dense path + reranker
translations      = {}  # qid -> canonicalize(german_translation) — used by dense path
bm25_query_texts  = {}  # qid -> f"{de_canon} {codes}"           — used by BM25 path

for qid, en_raw, de_raw in zip(all_query_ids, all_queries_en_raw, all_queries_de_raw):
    en_c = canonicalize(en_raw)
    de_c = canonicalize(de_raw)
    codes = extract_legal_codes(en_raw)  # extract from English original
    q_en_canon[qid]       = en_c
    translations[qid]     = de_c
    bm25_query_texts[qid] = f"{de_c} {codes}".strip()

# Sanity logs on first query
_sample_qid = all_query_ids[0]
print(f"  sample qid={_sample_qid}")
print(f"    EN canon: {q_en_canon[_sample_qid][:140]}")
print(f"    DE canon: {translations[_sample_qid][:140]}")
print(f"    BM25 txt: {bm25_query_texts[_sample_qid][:140]}")

# D-06: unload OpusMT BEFORE BGE-M3 loads. One model on GPU at a time.
_free_before = torch.cuda.mem_get_info()[0] / 1e9
print(f"  VRAM free before unload: {_free_before:.1f} GB")
unload(opus_tok, opus_mdl)
_free_after = torch.cuda.mem_get_info()[0] / 1e9
# Incomplete-unload guard (Pitfall 4): free VRAM must come back near the pre-load baseline.
# We conservatively require at least 10 GB free on T4 (16 GB total) after unload.
assert _free_after >= 10.0, (
    f"OpusMT unload incomplete: VRAM free={_free_after:.1f} GB (expected >= 10). "
    f"BGE-M3 load will OOM — investigate Pitfall 4."
)
print(f"  VRAM free after  unload: {_free_after:.1f} GB  [OK]")
'''


def main():
    nb = nbformat.read(NB_PATH, 4)

    # Dependency check: Cells 0-3 must already exist from Plan 01-01.
    if len(nb.cells) < 4:
        raise SystemExit(
            f"ABORT: notebook has only {len(nb.cells)} cells; Plan 01-01 "
            "Cells 0-3 are required before this plan can run."
        )
    # Sanity check that Cell 3 is the canonicalization helpers cell.
    cell3_src = nb.cells[3].source
    if "def canonicalize" not in cell3_src or "def extract_legal_codes" not in cell3_src:
        raise SystemExit(
            "ABORT: Cell 3 does not contain canonicalize/extract_legal_codes. "
            "Plan 01-01 must be applied first."
        )

    new_cell = nbformat.v4.new_code_cell(source=CELL_4_SRC)
    # Strip nbformat's auto-added execution_count/outputs keys that trip
    # validators when present with None values on a freshly built cell.
    new_cell.pop("id", None)

    if len(nb.cells) >= 5:
        # Replace in place — idempotent re-run or legacy-cell overwrite.
        nb.cells[4] = new_cell
    else:
        nb.cells.append(new_cell)

    nbformat.write(nb, NB_PATH)
    print(f"Wrote Cell 4 ({len(CELL_4_SRC)} chars) to {NB_PATH}")
    print(f"Notebook now has {len(nb.cells)} cells.")


if __name__ == "__main__":
    main()
