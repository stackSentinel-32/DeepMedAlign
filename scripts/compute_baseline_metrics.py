"""
compute_baseline_metrics.py
---------------------------
Computes Dice, HD95, and NCC for all registered subjects and saves
per-method CSV files to results/baseline_metrics_<method>.csv.

These numbers are the FLOOR.
VoxelMorph in Week 3 must beat Dice and HD95 to prove it has learned
meaningful deformations beyond classical registration.

Usage
-----
  python scripts/compute_baseline_metrics.py --method rigid
  python scripts/compute_baseline_metrics.py --method affine
  python scripts/compute_baseline_metrics.py --method bspline
  python scripts/compute_baseline_metrics.py --method affine --split val
"""

import sys
import argparse
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC, RESULTS, MANIFEST_P
from src.metrics import compute_all_metrics
from src.utils import get_logger, ensure_dir

log = get_logger("baseline_metrics")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute Dice/HD95/NCC for all registered subjects."
    )
    ap.add_argument(
        "--method", default="affine",
        choices=["rigid", "affine", "bspline"],
        help="Which registration output to evaluate.",
    )
    ap.add_argument(
        "--split", default=None,
        choices=["train", "val", "test"],
        help="Evaluate only one split.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N subjects.",
    )
    args = ap.parse_args()

    # --- Load manifest ---
    manifest_path = MANIFEST_P
    if not manifest_path.exists():
        log.error("manifest_processed.csv not found.")
        sys.exit(1)
    manifest = pd.read_csv(manifest_path)
    if args.split:
        manifest = manifest[manifest["split"] == args.split]
    if args.limit:
        manifest = manifest.head(args.limit)

    log.info(f"Evaluating {args.method} on {len(manifest)} subjects...")

    rows    = []
    t_start = time.time()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc=f"Metrics ({args.method})"):
        sid     = row["subject_id"]
        out     = DATA_PROC / sid
        mr_path = str(out / f"{sid}_mr_norm.nii.gz")
        ct_path = str(out / f"{sid}_ct_{args.method}.nii.gz")
        mr_msk  = str(out / f"{sid}_mr_brain_mask.nii.gz")
        ct_msk  = str(out / f"{sid}_ct_mask.nii.gz")

        if not Path(ct_path).exists():
            log.warning(f"{sid}: {args.method} output not found — skipping.")
            rows.append({
                "subject_id": sid,
                "split":      row.get("split", ""),
                "method":     args.method,
                "status":     "missing_registration",
            })
            continue

        try:
            m = compute_all_metrics(
                mr_path, ct_path, mr_msk, ct_msk,
                method=args.method,
            )
            rows.append({
                "subject_id": sid,
                "split":      row.get("split", ""),
                "status":     "ok",
                **m,
            })
        except Exception as exc:
            log.error(f"{sid}: {exc}")
            rows.append({
                "subject_id": sid,
                "split":      row.get("split", ""),
                "status":     f"error: {exc}",
            })

    results = pd.DataFrame(rows)
    ensure_dir(RESULTS)

    out_csv = RESULTS / f"baseline_metrics_{args.method}.csv"
    results.to_csv(out_csv, index=False)
    log.info(f"Saved: {out_csv}")

    # --- Summary table ---
    ok      = results[results["status"] == "ok"]
    elapsed = time.time() - t_start

    print()
    print("=" * 65)
    print(f"BASELINE METRICS  |  Method: {args.method.upper()}")
    print("=" * 65)

    for metric, label, direction in [
        ("dice", "Dice (brain-mask overlap)", "higher is better"),
        ("hd95", "HD95 in mm",               "lower  is better"),
        ("ncc",  "NCC (secondary)",           "higher is better"),
    ]:
        if metric not in ok.columns:
            continue
        vals = ok[metric].dropna()
        if vals.empty:
            print(f"\n  {label}: no values computed (CT mask missing?)")
            continue
        print(f"\n  {label} ({direction}):")
        print(f"    Mean +/- Std : {vals.mean():.4f} +/- {vals.std():.4f}")
        print(f"    Min  / Max   : {vals.min():.4f} / {vals.max():.4f}")
        print(f"    Median       : {vals.median():.4f}")

    print(f"\n  Subjects OK  : {len(ok)} / {len(results)}")
    print(f"  Wall time    : {elapsed:.1f}s")
    print(f"  Output CSV   : {out_csv}")
    print()
    print("  *** THESE ARE YOUR FLOOR NUMBERS ***")
    print("  VoxelMorph (Week 3) must beat Dice and HD95 to be useful.")
    print("=" * 65)


if __name__ == "__main__":
    main()
