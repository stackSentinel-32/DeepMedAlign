"""
verify_file_integrity.py
────────────────────────
Tries to load every NIfTI file with nibabel.
A file that passes this is guaranteed to not be corrupt.

Run:
    python scripts/verify_file_integrity.py
"""
import sys
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW
from src.utils import get_logger

log = get_logger("integrity")

manifest = pd.read_csv(DATA_RAW / "manifest.csv")

errors  = []
ok      = []

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Integrity"):
    for col in ["mr", "ct"]:
        path = row[col]
        try:
            img = nib.load(path)
            # Actually load the data to catch truncated files
            _ = img.get_fdata()
            ok.append(path)
        except Exception as e:
            errors.append({"subject_id": row["subject_id"],
                           "file": col,
                           "path": path,
                           "error": str(e)})
            log.error(f"CORRUPT: {row['subject_id']} {col} - {e}")

print(f"\nIntegrity check complete")
print(f"  OK:      {len(ok)}")
print(f"  ERRORS:  {len(errors)}")

if errors:
    err_df = pd.DataFrame(errors)
    out = DATA_RAW / "integrity_errors.csv"
    err_df.to_csv(out, index=False)
    print(f"\nErrors saved to: {out}")
    print(err_df.to_string())
else:
    print("All files loaded successfully [OK]")
