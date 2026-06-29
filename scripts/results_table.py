"""
results_table.py
----------------
Generates the master comparison table for all methods evaluated so far.
This table is updated each week as new methods are added.

Week 2: rigid, affine, bspline (classical baseline)
Week 3: voxelmorph (deep learning)
Week 4+: further methods

Run:
  python scripts/results_table.py
  python scripts/results_table.py --split test
  python scripts/results_table.py --out results/final_table.csv
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS
from src.utils import get_logger

log = get_logger("results_table")

# Methods to include (in display order)
METHODS = ["rigid", "affine", "bspline", "voxelmorph"]

# Metrics to include in the table
METRIC_COLS = ["dice", "hd95", "ncc"]


def _load_method(method: str, split: str = None) -> pd.DataFrame:
    """Load a single method's baseline CSV and return a summary row."""
    csv = RESULTS / f"baseline_metrics_{method}.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)
    ok = df[df["status"] == "ok"]
    if split:
        ok = ok[ok["split"] == split]
    if ok.empty:
        return None

    row = {"method": method, "n_subjects": len(ok)}
    for col in METRIC_COLS:
        if col in ok.columns:
            vals = ok[col].dropna()
            if not vals.empty:
                row[f"{col}_mean"] = round(vals.mean(), 4)
                row[f"{col}_std"]  = round(vals.std(),  4)
                row[f"{col}_med"]  = round(vals.median(), 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print master results comparison table."
    )
    ap.add_argument("--split", default=None,
                    choices=["train", "val", "test"],
                    help="Report only for one split.")
    ap.add_argument("--out", default=None,
                    help="Optional: path to save the table as CSV.")
    args = ap.parse_args()

    rows = []
    for method in METHODS:
        row = _load_method(method, split=args.split)
        if row is None:
            log.info(f"  {method}: no results found — skipping.")
        else:
            rows.append(row)

    if not rows:
        print("No baseline CSVs found. Run compute_baseline_metrics.py first.")
        return

    table = pd.DataFrame(rows)

    split_label = f" (split={args.split})" if args.split else " (all splits)"

    print()
    print("=" * 75)
    print(f"DEEPMEDALIGN — METHOD COMPARISON TABLE{split_label}")
    print("=" * 75)
    print(table.to_string(index=False))
    print()
    print("  Dice : higher is better  (> 0.85 = good for affine)")
    print("  HD95 : lower  is better  (< 5 mm = good for affine)")
    print("  NCC  : secondary metric only (MRI-CT is not linear)")
    print("=" * 75)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
