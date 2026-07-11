"""
evaluate_voxelmorph.py
-----------------------
Evaluates the trained VoxelMorph model on the test set.

What this script does:
  1. Loads `models/voxelmorph_best.pth`
  2. For each test subject:
     - Warps the CT using the VoxelMorph model.
     - Warps the CT brain mask using the predicted DVF (nearest neighbor).
     - Saves the warped CT and warped CT mask as .nii.gz files.
     - Computes Dice, HD95, and NCC against the MRI using `src.metrics`.
  3. Saves the final results to `results/baseline_metrics_voxelmorph.csv`.

Usage:
  python scripts/evaluate_voxelmorph.py
"""

import sys
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PROC, RESULTS, MANIFEST_P, MODELS
from src.voxelmorph_model import VoxelMorph, SpatialTransformer
from src.metrics import compute_all_metrics
from src.utils import get_logger, ensure_dir

log = get_logger("eval_voxelmorph")

def save_nifti(tensor, reference_nii_path, save_path, is_mask=False):
    """Convert tensor to numpy and save as NIfTI using a reference affine."""
    arr = tensor.detach().cpu().squeeze().numpy()
    
    if is_mask:
        arr = (arr > 0.5).astype(np.uint8)
        
    ref_nii = nib.load(reference_nii_path)
    out_nii = nib.Nifti1Image(arr, affine=ref_nii.affine, header=ref_nii.header)
    nib.save(out_nii, save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "voxelmorph_best.pth"), help="Path to trained model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # --- Load manifest ---
    if not MANIFEST_P.exists():
        log.error("manifest_processed.csv not found.")
        sys.exit(1)
        
    manifest = pd.read_csv(MANIFEST_P)
    manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    
    if len(manifest) == 0:
        log.error("No test subjects found in manifest.")
        sys.exit(1)

    log.info(f"Evaluating VoxelMorph on {len(manifest)} test subjects...")

    # --- Load Model ---
    device = torch.device(args.device)
    model = VoxelMorph().to(device)
    
    if not Path(args.model).exists():
        log.error(f"Model weights not found at {args.model}")
        sys.exit(1)
        
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded model weights from Epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.4f})")

    # Mask spatial transformer (nearest neighbor interpolation to prevent soft masks)
    mask_transformer = SpatialTransformer(size=(160, 192, 160), mode="nearest").to(device)

    rows = []
    t_start = time.time()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="VoxelMorph Evaluation"):
        sid = row["subject_id"]
        out = DATA_PROC / sid
        
        # Input paths
        mr_path      = str(out / f"{sid}_mr_norm.nii.gz")
        ct_path      = str(out / f"{sid}_ct_norm.nii.gz")
        mr_mask_path = str(out / f"{sid}_mr_brain_mask.nii.gz")
        ct_mask_path = str(out / f"{sid}_ct_mask.nii.gz")
        
        # Output paths
        warped_ct_path   = str(out / f"{sid}_ct_voxelmorph.nii.gz")
        warped_mask_path = str(out / f"{sid}_ct_mask_voxelmorph.nii.gz")
        
        if not (Path(mr_path).exists() and Path(ct_path).exists()):
            log.warning(f"Inputs missing for {sid} — skipping.")
            continue

        try:
            # 1. Load data as tensors (add batch and channel dims)
            mr_img = nib.load(mr_path).get_fdata().astype("float32")
            ct_img = nib.load(ct_path).get_fdata().astype("float32")
            ct_mask = nib.load(ct_mask_path).get_fdata().astype("float32")
            
            mr_t = torch.from_numpy(mr_img).unsqueeze(0).unsqueeze(0).to(device)
            ct_t = torch.from_numpy(ct_img).unsqueeze(0).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(ct_mask).unsqueeze(0).unsqueeze(0).to(device)

            # 2. Forward pass
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                warped_ct, dvf = model(mr_t, ct_t)
                warped_mask = mask_transformer(mask_t, dvf)

            # 3. Save warped CT and warped Mask as NIfTI
            save_nifti(warped_ct, mr_path, warped_ct_path)
            save_nifti(warped_mask, mr_path, warped_mask_path, is_mask=True)

            # 4. Compute standard metrics
            m = compute_all_metrics(
                mr_path, 
                warped_ct_path, 
                mr_mask_path, 
                warped_mask_path,
                method="voxelmorph",
            )
            
            rows.append({
                "subject_id": sid,
                "split": "test",
                "status": "ok",
                **m,
            })
            
        except Exception as exc:
            log.error(f"{sid}: {exc}")
            rows.append({
                "subject_id": sid,
                "split": "test",
                "status": f"error: {exc}",
            })

    results = pd.DataFrame(rows)
    ensure_dir(RESULTS)
    
    out_csv = RESULTS / "baseline_metrics_voxelmorph.csv"
    results.to_csv(out_csv, index=False)
    log.info(f"Saved: {out_csv}")

    # --- Summary table ---
    ok = results[results["status"] == "ok"]
    elapsed = time.time() - t_start

    print("\n" + "=" * 65)
    print("FINAL EVALUATION  |  Method: VOXELMORPH")
    print("=" * 65)

    for metric, label, direction in [
        ("dice", "Dice (brain-mask overlap)", "higher is better (target > 0.85)"),
        ("hd95", "HD95 in mm",                "lower is better (target < 5.0)"),
        ("ncc",  "NCC (secondary)",           "higher is better"),
    ]:
        if metric not in ok.columns: continue
        vals = ok[metric].dropna()
        if vals.empty: continue
        
        print(f"\n  {label}:")
        print(f"    Mean +/- Std : {vals.mean():.4f} +/- {vals.std():.4f}")
        print(f"    Median       : {vals.median():.4f}")

    print(f"\n  Subjects OK  : {len(ok)} / {len(results)}")
    print(f"  Wall time    : {elapsed:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()
