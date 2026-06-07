"""
Merge QC flags from shape_report.csv back into manifest.csv.
Produces manifest_v2.csv with a qc_flag column.
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW
from src.utils import get_logger

log = get_logger("update_manifest")

manifest = pd.read_csv(DATA_RAW / "manifest.csv")
report   = pd.read_csv(DATA_RAW / "shape_report.csv")

# Merge on subject_id
merged = manifest.merge(
    report[["subject_id", "flags", "status",
            "mr_shape", "ct_shape", "shapes_match",
            "ct_min_hu", "ct_max_hu",
            "mr_min", "mr_max"]],
    on="subject_id",
    how="left"
)

# Rename for clarity
merged.rename(columns={
    "flags":  "qc_flags",
    "status": "qc_status",
}, inplace=True)

out = DATA_RAW / "manifest_v2.csv"
merged.to_csv(out, index=False)
log.info(f"Saved: {out}")

print(f"\nManifest v2 columns: {list(merged.columns)}")
print(f"\nQC status breakdown:")
print(merged["qc_status"].value_counts())
print(f"\nSubjects with issues:")
bad = merged[merged["qc_status"] != "OK"]
if len(bad) == 0:
    print("  None - all clean.")
else:
    print(bad[["subject_id","split","qc_flags"]].to_string())
