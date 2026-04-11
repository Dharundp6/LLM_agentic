"""
Idempotent nbformat editor that installs Cell 13 in notebook_kaggle.ipynb.

Cell 13 = Pipeline completion summary + smoke-test exit banner. SMOKE gating
is applied inline in Cells 6 (laws truncation) and 7 (val truncation), so by
the time control reaches Cell 13 the end-to-end pipeline has already run.

Plan: 01-foundation-laws-pipeline / 01-05 Task 3
Depends on Plan 01-05 Tasks 1-2 having installed Cells 9-12.

Running this script is safe on either starting state:
  * if notebook has exactly 13 cells (0-12): Cell 13 is appended.
  * if notebook already has >=14 cells: Cell 13 is replaced in place.
"""
from __future__ import annotations

from pathlib import Path
import sys

import nbformat


NOTEBOOK = Path("notebook_kaggle.ipynb")
CELL13_INDEX = 13

CELL13_SOURCE = '''# Cell 13 — Pipeline completion summary + smoke-test exit banner.
# SMOKE gating is applied inline in Cells 6 (laws truncation) and 7 (val truncation),
# so by the time we reach this cell the pipeline has already completed end-to-end.

print(f"\\n=== Pipeline complete ===")
print(f"  Mode: {'SMOKE' if SMOKE else 'FULL'}")
print(f"  Val macro-F1: {best_f1:.4f}  @  top-{best_k}")
print(f"  submission.csv rows: {len(submission_df)}")
print(f"  submission.csv path: {OUT_PATH}")
print(f"  Total wall-clock elapsed since Cell 1: (log manually via the cell timings above)")

if SMOKE:
    print("\\nSMOKE mode passed end-to-end. Now unset SMOKE env var and re-run for full inference.")
else:
    print("\\nFULL mode complete. Next: human-verify checkpoint, then `git tag phase-1-submission`.")
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

    # Dependency check: Plan 01-05 Tasks 1-2 must have installed Cells 9-12.
    if len(nb.cells) < 13:
        print(
            f"ERROR: notebook has only {len(nb.cells)} cells; expected >=13 from Plan 01-05 Tasks 1-2",
            file=sys.stderr,
        )
        return 2
    if "submission_df.to_csv(OUT_PATH" not in nb.cells[12].source:
        print(
            "ERROR: Cell 12 missing submission_df.to_csv(OUT_PATH — Plan 01-05 Task 2 not applied",
            file=sys.stderr,
        )
        return 3
    if "Val macro-F1" not in nb.cells[11].source:
        print(
            "ERROR: Cell 11 missing Val macro-F1 banner — Plan 01-05 Task 2 not applied",
            file=sys.stderr,
        )
        return 4

    _install(nb, CELL13_INDEX, CELL13_SOURCE)

    nbformat.write(nb, NOTEBOOK)
    print(f"OK: notebook now has {len(nb.cells)} cells; Cell {CELL13_INDEX} installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
