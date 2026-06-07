"""
build_manifest.py
─────────────────
Scans the SynthRAD2023 brain directory, discovers all subjects,
validates that required files exist, and produces a manifest CSV
with patient-level train/val/test splits.

Run:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --root data/raw/synthrad/brain --out data/raw/manifest.csv
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SYNTHRAD, DATA_RAW, RANDOM_SEED
from src.utils import get_logger, ensure_dir

log = get_logger("build_manifest")


REQUIRED_FILES = ["mr.nii.gz", "ct.nii.gz"]
OPTIONAL_FILES = ["mask.nii.gz"]


def scan_subjects(root: Path) -> pd.DataFrame:
    """Walk root directory and collect one row per valid subject."""
    if not root.exists():
        log.error(f"Root does not exist: {root}")
        log.error("Have you downloaded SynthRAD2023 yet?")
        return pd.DataFrame()

    rows = []
    skipped = []

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue

        # Check required files
        missing = [f for f in REQUIRED_FILES
                   if not (subj_dir / f).exists()]
        if missing:
            log.warning(f"Skipping {subj_dir.name} — missing: {missing}")
            skipped.append({"subject_id": subj_dir.name,
                            "reason": f"missing {missing}"})
            continue

        row = {
            "subject_id": subj_dir.name,
            "mr":  str(subj_dir / "mr.nii.gz"),
            "ct":  str(subj_dir / "ct.nii.gz"),
        }

        # Optional files
        for f in OPTIONAL_FILES:
            key = f.replace(".nii.gz", "")
            row[key]            = str(subj_dir / f) if (subj_dir / f).exists() else ""
            row[f"has_{key}"]   = (subj_dir / f).exists()

        rows.append(row)

    if skipped:
        log.warning(f"Skipped {len(skipped)} subjects — see logs/r1/skipped.csv")
        pd.DataFrame(skipped).to_csv("logs/r1/skipped.csv", index=False)

    log.info(f"Found {len(rows)} complete subjects in {root}")
    return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame,
                train_frac: float = 0.70,
                val_frac:   float = 0.10,
                test_frac:  float = 0.20,
                seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Patient-level train/val/test split.
    NEVER split by slice — that causes catastrophic data leakage.
    Every slice from one subject MUST stay in one split.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_frac, random_state=seed, shuffle=True)

    # Second split: train vs val
    val_size_adjusted = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted,
        random_state=seed, shuffle=True)

    train["split"] = "train"
    val["split"]   = "val"
    test["split"]  = "test"

    result = pd.concat([train, val, test]).reset_index(drop=True)
    return result


def print_summary(df: pd.DataFrame):
    """Print a readable summary to terminal."""
    print("\n" + "="*50)
    print("MANIFEST SUMMARY")
    print("="*50)
    print(f"Total subjects : {len(df)}")
    print(f"\nSplit breakdown:")
    for split, grp in df.groupby("split"):
        print(f"  {split:6s}: {len(grp):4d} subjects")
    if "has_mask" in df.columns:
        n_mask = df["has_mask"].sum()
        print(f"\nSubjects with mask.nii.gz: {n_mask}/{len(df)}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build SynthRAD manifest CSV")
    parser.add_argument("--root", type=str, default=str(SYNTHRAD),
                        help="Path to SynthRAD brain directory")
    parser.add_argument("--out", type=str, default=str(DATA_RAW / "manifest.csv"),
                        help="Output manifest CSV path")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.10)
    parser.add_argument("--test",  type=float, default=0.20)
    args = parser.parse_args()

    ensure_dir("logs/r1")
    root = Path(args.root)
    out  = Path(args.out)

    log.info(f"Scanning: {root}")
    df = scan_subjects(root)

    if df.empty:
        log.error("No subjects found. Check your data path.")
        log.error(f"Expected structure: {root}/<subject_id>/mr.nii.gz")
        sys.exit(1)

    log.info("Creating splits...")
    df = make_splits(df, args.train, args.val, args.test)

    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    log.info(f"Manifest saved: {out}")

    print_summary(df)
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")


if __name__ == "__main__":
    main()
