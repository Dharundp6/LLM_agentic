"""
Build Cells 2 and 3 of notebook_kaggle.ipynb for Phase 1 Plan 01-01.

Cell 2 — Load laws_de.csv / val.csv / test.csv; log shapes + memory; A1-A10
corpus inspection banner.

Cell 3 — canonicalize() / extract_legal_codes() / tokenize_for_bm25_de()
helpers, LAWS-05 canonicalization of laws corpus, non-collapsing assert
(Pitfall 6 guard).

Idempotent: replaces cells [2] and [3] in place. Cells 0-1 (from the prior
task) and any trailing cells are untouched.
"""
from pathlib import Path
import nbformat

NB_PATH = Path(__file__).resolve().parent.parent / "notebook_kaggle.ipynb"

CELL_2_SRC = '''# Cell 2 — Load laws/val/test CSVs, log shapes, canonicalize laws.

assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"
t0 = time.time()

laws = pd.read_csv(DATA_DIR / "laws_de.csv").fillna("")
val  = pd.read_csv(DATA_DIR / "val.csv").fillna("")
test = pd.read_csv(DATA_DIR / "test.csv").fillna("")

print(f"Loaded CSVs in {time.time()-t0:.1f}s")
print(f"  laws : {laws.shape}  columns={list(laws.columns)}  "
      f"mem={laws.memory_usage(deep=True).sum()/1e6:.0f} MB")
print(f"  val  : {val.shape}   columns={list(val.columns)}")
print(f"  test : {test.shape}  columns={list(test.columns)}")

# A1-A10 data inspection banner for the Kaggle reviewer
print("\\n=== Corpus inspection ===")
print(f"laws['citation'] unique count: {laws['citation'].nunique():,}")
print(f"laws['citation'] sample: {laws['citation'].head(5).tolist()}")
print(f"val query count: {len(val)}")
print(f"test query count: {len(test)}")
'''

CELL_3_SRC = '''# Cell 3 — Canonicalization and tokenization helpers.
# canonicalize() is the single source of truth for query, corpus, and submission
# citation formatting (D-05, LAWS-05, QUERY-04, CALIB-02).

LAW_CODE_ALIASES = {
    "CC":  "ZGB",   # Code civil -> Zivilgesetzbuch
    "CO":  "OR",    # Code des obligations -> Obligationenrecht
    "CP":  "StGB",  # Code penal -> Strafgesetzbuch
    "CPP": "StPO",  # Code de procedure penale -> Strafprozessordnung
    "LP":  "SchKG", # Loi sur la poursuite -> SchKG
    "LTF": "BGG",   # Loi sur le Tribunal federal -> BGG
}

def canonicalize(s):
    """
    Canonicalize to Swiss German canonical form used in laws_de.csv.
    Idempotent. Rules (from 01-RESEARCH.md section Swiss Citation Canonicalization):
      1. Strip + collapse whitespace to single space.
      2. sharp-s (U+00DF) -> 'ss' (Swiss German standard).
      3. 'Art.X' / 'Art.  X' -> 'Art. X' (single space). Same for Abs./lit./Ziff.
      4. Map French-language law code aliases to German (CC->ZGB etc.).
      5. Preserve BGE Roman numerals and 'E.' vs 'E ' variants as-is.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("\\u00df", "ss")  # sharp-s -> ss
    s = re.sub(r"\\s+", " ", s)
    s = re.sub(r"\\b(Art|Abs|lit|Ziff)\\.\\s*(\\d)", r"\\1. \\2", s)
    for _fr, _de in LAW_CODE_ALIASES.items():
        s = re.sub(rf"\\b{_fr}\\b", _de, s)
    return s

# Idempotence smoke check (fail-loud)
assert canonicalize("Art.11 Abs.2 CC") == "Art. 11 Abs. 2 ZGB", \\
    f"canonicalize broken: {canonicalize('Art.11 Abs.2 CC')!r}"
assert canonicalize(canonicalize("Art.11 Abs.  2 CC")) == canonicalize("Art.11 Abs.  2 CC")

# --- extract_legal_codes: reused from solution.py:319-330 for D-03 ---
QUERY_CODES = {
    "ZGB", "OR", "StGB", "StPO", "BGG", "SchKG", "BGB",
    "AHVG", "IVG", "ELG", "KVG", "UVG", "BVG",
    "DBG", "MWSTG", "USG", "RPG",
}

def extract_legal_codes(text):
    """
    Pull '(law_code, article_number)' tuples from the ORIGINAL English query
    text for D-03. Returns a space-joined string of legal code tokens to
    append to the German translation before BM25 tokenization.
    """
    if not isinstance(text, str):
        return ""
    out = []
    # Art. N CODE patterns (e.g., "Art. 41 OR")
    for m in re.finditer(r"\\bArt\\.\\s*(\\d+[a-z]?)\\s*([A-Z][A-Za-z]{1,6})\\b", text):
        num, code = m.group(1), m.group(2)
        if code in QUERY_CODES:
            out.append(f"Art. {num} {code}")
    # Bare law code tokens (e.g., "under OR", "ZGB art. 1")
    for m in re.finditer(r"\\b([A-Z][A-Za-z]{1,6})\\b", text):
        if m.group(1) in QUERY_CODES:
            out.append(m.group(1))
    # BGE references
    for m in re.finditer(r"\\bBGE\\s+\\d+\\s+[IVX]+[a-z]?\\s+\\d+\\b", text):
        out.append(m.group(0))
    return " ".join(out)

# --- tokenize_for_bm25_de: legal-aware German tokenizer for bm25s ---
# Full CharSplit decompounding wiring and NLTK stopwords happen here. bm25s
# itself just consumes the list-of-list tokens we produce.
from nltk.corpus import stopwords as _nltk_stopwords
GERMAN_STOP = frozenset(_nltk_stopwords.words("german"))

try:
    from charsplit import Splitter
    _CHARSPLIT = Splitter()
    _CHARSPLIT_OK = True
except Exception as _e:
    print(f"  CharSplit unavailable ({_e}); decompounding disabled.", flush=True)
    _CHARSPLIT = None
    _CHARSPLIT_OK = False

def tokenize_for_bm25_de(text):
    """
    Legal-aware German tokenization for bm25s input.
      1. Lowercase; split on word+dot boundaries (preserves Art., Abs., BGE, E., numbers).
      2. Remove German stopwords (der, die, das, gemaess, ...).
      3. For tokens longer than 3 chars without a period, attempt CharSplit
         decompounding and APPEND (not replace) the head/tail split if score > 0.5.
    Returns list[str].
    """
    text = str(text).lower()
    raw = re.findall(r"[\\w.]+", text)
    out = []
    for tok in raw:
        if tok in GERMAN_STOP:
            continue
        if len(tok) <= 3 or "." in tok:
            out.append(tok)
            continue
        out.append(tok)
        if _CHARSPLIT_OK:
            try:
                splits = _CHARSPLIT.split_compound(tok)
                if splits:
                    score, head, tail = splits[0]
                    if score > 0.5 and head and tail:
                        out.append(head.lower())
                        out.append(tail.lower())
            except Exception:
                pass
    return out

# Legal-notation survival smoke check
_sample_tokens = tokenize_for_bm25_de("Die Pflicht aus Art. 41 OR")
assert "art." in _sample_tokens, f"Art. dropped: {_sample_tokens}"
assert "or" in _sample_tokens, f"OR dropped: {_sample_tokens}"

# === LAWS-05: apply canonicalize to laws corpus BEFORE any index build ===
t0 = time.time()
laws["citation"] = laws["citation"].apply(canonicalize)
laws["text"]     = laws["text"].apply(canonicalize)
print(f"Canonicalized laws corpus in {time.time()-t0:.1f}s")

# === Pitfall 6 guard: non-collapsing assertion ===
# canonicalize() must NOT collapse two distinct corpus citations into the same string.
_raw_unique = laws["citation"].drop_duplicates()
assert len(set(canonicalize(c) for c in _raw_unique)) == len(set(_raw_unique)), (
    "canonicalize() collapses distinct corpus citations — shrink LAW_CODE_ALIASES"
)
print(f"Non-collapsing canonicalize assert OK on {len(_raw_unique):,} unique citations")
'''


def main():
    nb = nbformat.read(str(NB_PATH), as_version=4)

    cell2 = nbformat.v4.new_code_cell(source=CELL_2_SRC)
    cell3 = nbformat.v4.new_code_cell(source=CELL_3_SRC)

    while len(nb.cells) < 4:
        nb.cells.append(nbformat.v4.new_code_cell(source=""))

    nb.cells[2] = cell2
    nb.cells[3] = cell3

    nbformat.write(nb, str(NB_PATH))
    print(f"Wrote Cells 2 and 3 to {NB_PATH}")
    print(f"Total cells now: {len(nb.cells)}")


if __name__ == "__main__":
    main()
