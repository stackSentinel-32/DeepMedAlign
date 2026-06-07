"""
dataset_summary.py
──────────────────
Generates a readable summary table of the full dataset
for documentation and the README results table.

Run:
    python scripts/dataset_summary.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW

report   = pd.read_csv(DATA_RAW / "shape_report.csv")
manifest = pd.read_csv(DATA_RAW / "manifest.csv")

# ── Overall stats ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("DEEPMEDALIGN - DATASET SUMMARY")
print("="*65)

print(f"\n{'Dataset':25s}: SynthRAD2023 Task 1 - Brain")
print(f"{'Total subjects':25s}: {len(manifest)}")
print(f"{'With mask.nii.gz':25s}: {manifest['has_mask'].sum() if 'has_mask' in manifest else 'N/A'}")

# ── Split breakdown ────────────────────────────────────────────────────────────
print("\n--- Split Breakdown ---")
for split in ["train", "val", "test"]:
    n = len(manifest[manifest["split"] == split])
    print(f"  {split:8s}: {n:4d} subjects")

# ── Shape statistics ──────────────────────────────────────────────────────────
print("\n--- MRI Shape Distribution ---")
shapes = report["mr_shape"].value_counts().head(5)
for shape, count in shapes.items():
    print(f"  {shape}: {count} subjects")

# ── Intensity statistics ───────────────────────────────────────────────────────
print("\n--- CT HU Range Statistics ---")
print(f"  Min HU (across subjects): {report['ct_min_hu'].min():.0f}")
print(f"  Max HU (across subjects): {report['ct_max_hu'].max():.0f}")
print(f"  Mean of min HU:           {report['ct_min_hu'].mean():.1f}")
print(f"  Mean of max HU:           {report['ct_max_hu'].mean():.1f}")

print("\n--- MRI Intensity Statistics ---")
print(f"  Min (across subjects):  {report['mr_min'].min():.1f}")
print(f"  Max (across subjects):  {report['mr_max'].max():.1f}")
print(f"  Mean of mean:           {report['mr_mean'].mean():.1f}")

# ── QC Summary ────────────────────────────────────────────────────────────────
print("\n--- QC Summary ---")
ok_count   = len(report[report["flags"] == "OK"])
fail_count = len(report[report["flags"] != "OK"])
print(f"  Passed QC: {ok_count}/{len(report)}")
if fail_count > 0:
    print(f"  Failed QC: {fail_count} (see shape_report.csv for details)")

# ── Storage ────────────────────────────────────────────────────────────────────
if "mr_bytes" in report.columns:
    total_gb = (report["mr_bytes"].sum() + report.get("ct_bytes", pd.Series([0])).sum()) / 1e9
    print(f"\n--- Storage ---")
    print(f"  Total data size: {total_gb:.6f} GB (approx)")

print("="*65 + "\n")
