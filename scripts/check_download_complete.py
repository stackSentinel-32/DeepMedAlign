"""
check_download_complete.py
──────────────────────────
Checks which subjects are fully downloaded vs still missing.
Run this while the download is in progress to monitor status.

Run:
    python scripts/check_download_complete.py
    watch -n 30 python scripts/check_download_complete.py   # auto-refresh every 30s
"""
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SYNTHRAD

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Download Status Check")
print("="*50)

root = SYNTHRAD

if not root.exists():
    print(f"ERROR: {root} does not exist yet")
    sys.exit(1)

subject_dirs = [d for d in root.iterdir() if d.is_dir()]
print(f"Subject directories found: {len(subject_dirs)}")

complete   = []
incomplete = []

for subj in sorted(subject_dirs):
    mr   = (subj / "mr.nii.gz").exists()
    ct   = (subj / "ct.nii.gz").exists()
    mask = (subj / "mask.nii.gz").exists()

    if mr and ct:
        complete.append(subj.name)
    else:
        missing = []
        if not mr:   missing.append("mr.nii.gz")
        if not ct:   missing.append("ct.nii.gz")
        incomplete.append((subj.name, missing))

print(f"Complete (MR+CT):  {len(complete)}")
print(f"Incomplete:        {len(incomplete)}")

if incomplete:
    print("\nIncomplete subjects:")
    for sid, missing in incomplete[:10]:
        print(f"  {sid}: missing {missing}")
    if len(incomplete) > 10:
        print(f"  ... and {len(incomplete)-10} more")

# Total size
total_bytes = 0
for subj in subject_dirs:
    for f in subj.glob("*.nii.gz"):
        total_bytes += f.stat().st_size

print(f"\nTotal downloaded: {total_bytes/1e9:.6f} GB")
print("="*50)
