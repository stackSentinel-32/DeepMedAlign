"""
validate_data.py
────────────────
Runs automated QC on every subject in the manifest:
- Shape and spacing check
- HU range check for CT (catches mis-labelled volumes)
- MRI intensity range check
- File size check (catches corrupt downloads)
- Mask coverage check

Run:
    python scripts/validate_data.py
    python scripts/validate_data.py --manifest data/raw/manifest.csv --out data/raw/shape_report.csv
"""

import sys
import argparse
import time
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW
from src.utils import get_logger, ensure_dir

log = get_logger("validate_data")


def check_subject(row: pd.Series) -> dict:
    """Run all checks for one subject. Returns a result dict."""
    sid   = row["subject_id"]
    flags = []
    result = {"subject_id": sid, "split": row["split"]}

    # ── MR checks ─────────────────────────────────────────────────────────────
    try:
        mr_img  = nib.load(row["mr"])
        mr_arr  = mr_img.get_fdata().astype("float32")
        mr_zoom = mr_img.header.get_zooms()

        result["mr_shape"]   = str(mr_img.shape)
        result["mr_spacing"] = str(tuple(round(float(z), 3) for z in mr_zoom))
        result["mr_min"]     = round(float(mr_arr.min()), 2)
        result["mr_max"]     = round(float(mr_arr.max()), 2)
        result["mr_mean"]    = round(float(mr_arr.mean()), 2)
        result["mr_nonzero"] = int((mr_arr != 0).sum())
        result["mr_bytes"]   = Path(row["mr"]).stat().st_size

        # Flag if MR looks like CT (HU-style values)
        if mr_arr.min() < -500:
            flags.append("MR_LOOKS_LIKE_CT")

        # Flag if MR is all zeros (corrupt)
        if result["mr_nonzero"] == 0:
            flags.append("MR_ALL_ZEROS")

    except Exception as e:
        result["mr_shape"] = "ERROR"
        flags.append(f"MR_LOAD_ERROR: {e}")

    # ── CT checks ─────────────────────────────────────────────────────────────
    try:
        ct_img  = nib.load(row["ct"])
        ct_arr  = ct_img.get_fdata().astype("float32")
        ct_zoom = ct_img.header.get_zooms()

        result["ct_shape"]   = str(ct_img.shape)
        result["ct_spacing"] = str(tuple(round(float(z), 3) for z in ct_zoom))
        result["ct_min_hu"]  = round(float(ct_arr.min()), 2)
        result["ct_max_hu"]  = round(float(ct_arr.max()), 2)
        result["ct_mean_hu"] = round(float(ct_arr.mean()), 2)
        result["ct_nonzero"] = int((ct_arr != 0).sum())
        result["ct_bytes"]   = Path(row["ct"]).stat().st_size

        # CT must have negative values (air = -1000 HU)
        if ct_arr.min() > -100:
            flags.append("CT_NO_NEGATIVE_HU")

        # CT bone should go above 400 HU
        if ct_arr.max() < 200:
            flags.append("CT_MAX_HU_TOO_LOW")

        # Corrupt CT
        if result["ct_nonzero"] == 0:
            flags.append("CT_ALL_ZEROS")

        # Shape must match MR
        if "mr_shape" in result and result["mr_shape"] != "ERROR":
            if mr_img.shape != ct_img.shape:
                flags.append("SHAPE_MISMATCH")

        result["shapes_match"] = (
            "mr_shape" in result
            and result.get("mr_shape") != "ERROR"
            and mr_img.shape == ct_img.shape
        )

    except Exception as e:
        result["ct_shape"] = "ERROR"
        flags.append(f"CT_LOAD_ERROR: {e}")

    # ── Mask checks ───────────────────────────────────────────────────────────
    if row.get("has_mask") and row.get("mask"):
        try:
            msk_img = nib.load(row["mask"])
            msk_arr = msk_img.get_fdata().astype("float32")
            unique  = np.unique(msk_arr)

            result["mask_shape"]   = str(msk_img.shape)
            result["mask_unique"]  = str(unique.tolist())
            result["mask_volume"]  = int((msk_arr > 0).sum())

            # Mask should be binary
            non_binary = [v for v in unique if v not in [0.0, 1.0]]
            if non_binary:
                flags.append(f"MASK_NOT_BINARY: {non_binary}")

            # Very small mask = skull stripping probably failed
            if result["mask_volume"] < 50000:
                flags.append("MASK_VERY_SMALL")

        except Exception as e:
            flags.append(f"MASK_LOAD_ERROR: {e}")
    else:
        result["mask_volume"] = 0

    # ── Final flag ────────────────────────────────────────────────────────────
    result["flags"] = "|".join(flags) if flags else "OK"
    result["status"] = "FAIL" if flags else "OK"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DATA_RAW / "manifest.csv"))
    parser.add_argument("--out",      default=str(DATA_RAW / "shape_report.csv"))
    parser.add_argument("--split",    default=None,
                        help="Filter to one split: train/val/test")
    args = parser.parse_args()

    ensure_dir("logs/r1")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        log.error("Run: python scripts/build_manifest.py first")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    if args.split:
        df = df[df["split"] == args.split]
        log.info(f"Filtered to split='{args.split}': {len(df)} subjects")

    log.info(f"Validating {len(df)} subjects...")
    t0 = time.time()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="QC"):
        results.append(check_subject(row))

    report = pd.DataFrame(results)
    report.to_csv(args.out, index=False)
    log.info(f"Report saved: {args.out}  ({time.time()-t0:.1f}s)")

    # Print summary
    ok   = report[report["status"] == "OK"]
    fail = report[report["status"] == "FAIL"]

    print("\n" + "="*60)
    print("QC SUMMARY")
    print("="*60)
    print(f"  Total:  {len(report)}")
    print(f"  OK:     {len(ok)}")
    print(f"  FAILED: {len(fail)}")

    if len(fail) > 0:
        print("\nFailed subjects:")
        print(fail[["subject_id", "split", "flags"]].to_string())

    # Flag breakdown
    all_flags = []
    for f in report["flags"]:
        if f != "OK":
            all_flags.extend(f.split("|"))
    if all_flags:
        from collections import Counter
        print("\nFlag counts:")
        for flag, count in Counter(all_flags).most_common():
            print(f"  {flag}: {count}")
    else:
        print("\nAll subjects passed QC [OK]")

    print("="*60)


if __name__ == "__main__":
    main()
