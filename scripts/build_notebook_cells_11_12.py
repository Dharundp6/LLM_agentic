"""
Idempotent nbformat editor that installs Cells 11 and 12 in notebook_kaggle.ipynb.

Cell 11 = Stage 6a: macro_f1 + calibrate_top_k on val_results (val-only, CALIB-05).
Cell 12 = Stage 6b: validate_submission + apply best_k to test + write submission.csv.

Plan: 01-foundation-laws-pipeline / 01-05 Task 2
Depends on Plan 01-05 Task 1 having installed Cells 9-10 (reranked_laws_ids).

Running this script is safe on either starting state:
  * if notebook has exactly 11 cells (0-10): Cells 11 and 12 are appended.
  * if notebook already has >=13 cells: Cells 11 and 12 are replaced in place.
"""
from __future__ import annotations

from pathlib import Path
import sys

import nbformat


NOTEBOOK = Path("notebook_kaggle.ipynb")
CELL11_INDEX = 11
CELL12_INDEX = 12

CELL11_SOURCE = '''# Cell 11 — Stage 6a: val F1 + calibrate top-K (CALIB-01 / CALIB-04 / CALIB-05).
# macro_f1 and calibrate_top_k ported verbatim from solution.py:288-313 with
# k_max=60 (instead of solution.py's 80) to match the plan-level hyperparameter.
# CALIB-05: calibrate_top_k receives val_results ONLY — test is never touched here.

def macro_f1(gold_sets, pred_sets):
    """Per-query F1, averaged across queries."""
    if not gold_sets:
        return 0.0
    f1s = []
    for g, p in zip(gold_sets, pred_sets):
        if not g and not p:
            f1s.append(1.0); continue
        if not g or not p:
            f1s.append(0.0); continue
        tp = len(g & p)
        if tp == 0:
            f1s.append(0.0); continue
        prec = tp / len(p)
        rec  = tp / len(g)
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)

def calibrate_top_k(val_results, k_min=1, k_max=60):
    """Grid search best fixed top-K on val. Warn on boundary hit (CP-4)."""
    best_f1, best_k = 0.0, 20
    gold_sets = [r["gold"] for r in val_results]
    for k in range(k_min, k_max + 1):
        pred_sets = [set(r["ranked_citations"][:k]) for r in val_results]
        f1 = macro_f1(gold_sets, pred_sets)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    if best_k == k_min or best_k == k_max:
        print(f"  WARN: best_k={best_k} at boundary of [{k_min}, {k_max}]; "
              f"overfitting risk (CP-4)", flush=True)
    print(f"*** Val macro-F1 = {best_f1:.4f}  @  top-{best_k} ***")
    return best_k, best_f1

# Build val_results for calibration (CALIB-05: val only, never touches test).
print(f"\\n=== Stage 6a: Calibration ===")
val_results = []
for _, row in val.iterrows():
    qid = row["query_id"]
    if qid not in reranked_laws_ids:   # SMOKE mode may have truncated val
        continue
    gold = set(str(row["gold_citations"]).split(";")) if row["gold_citations"] else set()
    # Map row indices back to canonical citation strings (CALIB-02: canonicalize on read).
    ranked_citations = [
        canonicalize(laws.iloc[idx]["citation"])
        for idx in reranked_laws_ids[qid]
    ]
    val_results.append({
        "query_id": qid,
        "gold": gold,
        "ranked_citations": ranked_citations,
    })

best_k, best_f1 = calibrate_top_k(val_results, k_min=1, k_max=60)
'''

CELL12_SOURCE = '''# Cell 12 — Stage 6b: apply best_k to test, canonicalize, validate, write.
# CALIB-03: validate_submission() runs BEFORE df.to_csv() so a malformed
# submission fails loudly instead of silently clobbering the output file.

def validate_submission(df, test_df):
    """CALIB-03: fail loudly before writing an invalid submission.csv."""
    assert list(df.columns) == ["query_id", "predicted_citations"], (
        f"wrong columns: {list(df.columns)}"
    )
    assert len(df) == len(test_df), (
        f"row count mismatch: submission={len(df)} test={len(test_df)}"
    )
    assert set(df["query_id"]) == set(test_df["query_id"]), (
        "query_id set does not match test.csv"
    )
    assert df["query_id"].is_unique, "duplicate query_id in submission"
    assert not df["predicted_citations"].isna().any(), "NaN in predicted_citations"
    for i, v in df["predicted_citations"].items():
        assert isinstance(v, str), f"row {i}: not a string"
        assert not v.endswith(";"), f"row {i}: trailing semicolon"
        for cit in v.split(";"):
            if cit and "," in cit:
                print(f"  WARN row {i}: comma inside citation '{cit}'", flush=True)
    print(f"validate_submission OK: {len(df)} rows", flush=True)

print(f"\\n=== Stage 6b: Write submission.csv ===")
sub_rows = []
for _, row in test.iterrows():
    qid = row["query_id"]
    if qid not in reranked_laws_ids:
        # Should not happen in production run — SMOKE mode only truncates val.
        sub_rows.append({"query_id": qid, "predicted_citations": ""})
        continue
    top_k_ids = reranked_laws_ids[qid][:best_k]
    cits = [canonicalize(laws.iloc[idx]["citation"]) for idx in top_k_ids]
    # Dedupe preserving order
    seen = set()
    deduped = []
    for c in cits:
        if c and c not in seen:
            deduped.append(c); seen.add(c)
    sub_rows.append({
        "query_id": qid,
        "predicted_citations": ";".join(deduped),
    })

submission_df = pd.DataFrame(sub_rows)[["query_id", "predicted_citations"]]
validate_submission(submission_df, test)
submission_df.to_csv(OUT_PATH, index=False)
print(f"submission.csv written to: {OUT_PATH}  rows={len(submission_df)}")
'''


def _install(nb, index: int, source: str) -> None:
    cell = nbformat.v4.new_code_cell(source=source)
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

    # Dependency check: Plan 01-05 Task 1 must have installed Cells 9-10.
    if len(nb.cells) < 11:
        print(
            f"ERROR: notebook has only {len(nb.cells)} cells; expected >=11 from Plan 01-05 Task 1",
            file=sys.stderr,
        )
        return 2
    if "reranked_laws_ids" not in nb.cells[10].source:
        print(
            "ERROR: Cell 10 missing reranked_laws_ids — Plan 01-05 Task 1 not applied",
            file=sys.stderr,
        )
        return 3
    if "fused_laws_ids" not in nb.cells[9].source:
        print(
            "ERROR: Cell 9 missing fused_laws_ids — Plan 01-05 Task 1 not applied",
            file=sys.stderr,
        )
        return 4

    _install(nb, CELL11_INDEX, CELL11_SOURCE)
    _install(nb, CELL12_INDEX, CELL12_SOURCE)

    nbformat.write(nb, NOTEBOOK)
    print(f"OK: notebook now has {len(nb.cells)} cells; Cells 11 and 12 installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
