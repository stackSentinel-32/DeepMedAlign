"""
run_classical.py
----------------
Batch classical registration runner.
Reads the manifest, finds all preprocessed subjects, and runs the
rigid -> affine -> B-spline pipeline on each one.

Usage
-----
  python scripts/run_classical.py                   # all subjects, all stages
  python scripts/run_classical.py --no-bspline      # rigid + affine only (faster)
  python scripts/run_classical.py --limit 5         # first 5 subjects
  python scripts/run_classical.py --split train     # one split
  python scripts/run_classical.py --subj 1BA001     # single subject
  python scripts/run_classical.py --force           # re-register already-done subjects
"""

import sys
import time
import argparse
import traceback
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC
from src.classical_reg import register_full_pipeline
from src.utils import get_logger, ensure_dir

log = get_logger("run_classical")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> pd.DataFrame:
    """Return the best available manifest CSV."""
    for name in ("manifest_processed.csv", "manifest_final.csv", "manifest.csv"):
        p = DATA_RAW / name
        if p.exists():
            log.info(f"Using manifest: {p.name}")
            return pd.read_csv(p)
    log.error("No manifest found. Run scripts/build_manifest.py first.")
    sys.exit(1)


def _already_registered(out: Path, sid: str, run_bspline: bool) -> bool:
    """Return True if this subject's registration outputs already exist."""
    affine_ok  = (out / f"{sid}_ct_affine.nii.gz").exists()
    bspline_ok = (out / f"{sid}_ct_bspline.nii.gz").exists()
    if run_bspline:
        return affine_ok and bspline_ok
    return affine_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch classical MRI-CT registration.")
    ap.add_argument("--limit",      type=int, default=None,
                    help="Process only the first N subjects (useful for testing).")
    ap.add_argument("--split",      default=None, choices=["train", "val", "test"],
                    help="Process only one data split.")
    ap.add_argument("--subj",       default=None,
                    help="Process a single subject by ID.")
    ap.add_argument("--no-bspline", action="store_true",
                    help="Skip the B-spline stage (rigid + affine only).")
    ap.add_argument("--force",      action="store_true",
                    help="Re-register subjects whose outputs already exist.")
    args = ap.parse_args()

    run_bspline = not args.no_bspline
    manifest    = _load_manifest()

    # --- Apply filters ---
    if args.subj:
        manifest = manifest[manifest["subject_id"] == args.subj]
        if manifest.empty:
            log.error(f"Subject '{args.subj}' not found in manifest.")
            sys.exit(1)
    if args.split:
        manifest = manifest[manifest["split"] == args.split]
    if args.limit:
        manifest = manifest.head(args.limit)

    log.info(f"Subjects to process : {len(manifest)}")
    log.info(f"B-spline stage      : {'ON' if run_bspline else 'OFF'}")
    log.info(f"Force re-register   : {args.force}")

    rows    = []
    t_start = time.time()
    n_ok, n_skip, n_error = 0, 0, 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Registering"):
        sid     = row["subject_id"]
        out     = DATA_PROC / sid
        mr_path = str(out / f"{sid}_mr_norm.nii.gz")
        ct_path = str(out / f"{sid}_ct_norm.nii.gz")

        # Check preprocessed inputs exist
        if not Path(mr_path).exists():
            log.warning(f"{sid}: mr_norm.nii.gz missing — skipping.")
            n_skip += 1
            rows.append({"subject_id": sid, "reg_status": "skip_mr_missing"})
            continue
        if not Path(ct_path).exists():
            log.warning(f"{sid}: ct_norm.nii.gz missing — skipping.")
            n_skip += 1
            rows.append({"subject_id": sid, "reg_status": "skip_ct_missing"})
            continue

        # Skip already-done subjects unless --force
        if not args.force and _already_registered(out, sid, run_bspline):
            log.info(f"{sid}: already registered — skipping (use --force to redo).")
            n_skip += 1
            rows.append({"subject_id": sid, "reg_status": "skip_already_done"})
            continue

        # Run registration pipeline
        try:
            reg_result = register_full_pipeline(
                mr_path, ct_path, str(out), sid,
                run_bspline=run_bspline,
            )
            rows.append({
                "subject_id": sid,
                "split":      row.get("split", ""),
                "reg_status": "ok",
                **reg_result,
            })
            n_ok += 1
        except Exception as exc:
            log.error(f"{sid} FAILED: {exc}")
            log.debug(traceback.format_exc())
            rows.append({
                "subject_id": sid,
                "reg_status": f"error: {str(exc)[:120]}",
            })
            n_error += 1

    elapsed = time.time() - t_start

    # --- Save updated manifest ---
    result_df = pd.DataFrame(rows)
    out_csv   = DATA_RAW / "manifest_registered.csv"
    try:
        merged = manifest.merge(
            result_df[["subject_id", "reg_status"]],
            on="subject_id", how="left",
        )
        merged.to_csv(out_csv, index=False)
    except Exception:
        result_df.to_csv(out_csv, index=False)

    # --- Summary ---
    print()
    print("=" * 60)
    print("CLASSICAL REGISTRATION — COMPLETE")
    print("=" * 60)
    print(f"  OK      : {n_ok}")
    print(f"  Skipped : {n_skip}")
    print(f"  Errors  : {n_error}")
    if n_ok > 0:
        print(f"  Avg time: {elapsed / n_ok:.1f}s per subject")
    print(f"  Total   : {elapsed:.1f}s")
    print(f"  Manifest: {out_csv}")
    if n_error > 0:
        print("\n  Failed subjects:")
        for r in rows:
            if r.get("reg_status", "").startswith("error"):
                print(f"    {r['subject_id']}: {r['reg_status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
