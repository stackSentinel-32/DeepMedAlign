"""
finalize_manifest.py
─────────────────────
Produces the final manifest_processed.csv that R3's preprocessing
runner reads. Adds placeholder columns for preprocessed file paths
so R3 can fill them in as subjects are processed.

Run:
    python scripts/finalize_manifest.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC
from src.utils import get_logger

log = get_logger("finalize_manifest")

# Start from manifest_v2 if it exists, else manifest
if (DATA_RAW / "manifest_v2.csv").exists():
    df = pd.read_csv(DATA_RAW / "manifest_v2.csv")
    log.info("Using manifest_v2.csv as base")
else:
    df = pd.read_csv(DATA_RAW / "manifest.csv")
    log.info("Using manifest.csv as base")

# Add placeholder columns for R3 to fill in
for col in [
    "mr_preprocessed",
    "mr_mask",
    "ct_preprocessed",
    "ct_mask",
    "ct_affine_aligned",
    "affine_transform",
    "preprocess_status",
    "classical_reg_status",
]:
    if col not in df.columns:
        df[col] = ""

# Add the expected preprocessed paths (R3 will confirm these exist)
for idx, row in df.iterrows():
    sid = row["subject_id"]
    out = DATA_PROC / sid
    df.at[idx, "mr_preprocessed"]   = str(out / f"{sid}_mr_norm.nii.gz")
    df.at[idx, "mr_mask"]           = str(out / f"{sid}_mr_brain_mask.nii.gz")
    df.at[idx, "ct_preprocessed"]   = str(out / f"{sid}_ct_norm.nii.gz")
    df.at[idx, "ct_mask"]           = str(out / f"{sid}_ct_mask.nii.gz")
    df.at[idx, "ct_affine_aligned"] = str(out / f"{sid}_ct_affine.nii.gz")
    df.at[idx, "affine_transform"]  = str(out / f"{sid}_affine.tfm")

out_path = DATA_RAW / "manifest_final.csv"
df.to_csv(out_path, index=False)
log.info(f"Final manifest saved: {out_path}")

print(f"\nFinal manifest columns:")
for col in df.columns:
    print(f"  {col}")
print(f"\nTotal subjects: {len(df)}")
print(f"\nReady for R3 preprocessing pipeline.")
print(f"Path: {out_path}")
