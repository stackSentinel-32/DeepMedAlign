"""
evaluate_voxelmorph.py
-----------------------
Evaluates the trained VoxelMorph model on the test set.

Usage:
  python scripts/evaluate_voxelmorph.py
  python scripts/evaluate_voxelmorph.py --tta          # Test-Time Adaptation
  python scripts/evaluate_voxelmorph.py --diffeomorphic
"""

import sys
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PROC, RESULTS, MANIFEST_P, MODELS
from src.voxelmorph_model import VoxelMorph, SpatialTransformer
from src.losses import total_loss
from src.metrics import compute_all_metrics
from src.utils import get_logger, ensure_dir

log = get_logger("eval_voxelmorph")


def save_nifti(tensor, reference_nii_path, save_path, is_mask=False):
    arr = tensor.detach().cpu().squeeze().numpy()
    if is_mask:
        arr = (arr > 0.5).astype(np.uint8)
    ref_nii = nib.load(reference_nii_path)
    nib.save(nib.Nifti1Image(arr, ref_nii.affine, ref_nii.header), save_path)


def test_time_adapt(model, mr_t, ct_t, device, steps=30, lr=1e-4):
    """Fine-tune the model on a single subject for `steps` gradient steps.

    This is Test-Time Adaptation (TTA): squeezes extra accuracy out of the
    final model without changing the saved checkpoint.
    """
    adapted = type(model).__new__(type(model))
    adapted.__dict__.update(model.__dict__)
    adapted = model.__class__(
        diffeomorphic=model.diffeomorphic
    ).to(device)
    adapted.load_state_dict(model.state_dict())
    adapted.train()

    opt = optim.Adam(adapted.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        warped_ct, dvf = adapted(mr_t, ct_t)
        loss, _ = total_loss(warped_ct, mr_t, dvf)
        loss.backward()
        opt.step()

    adapted.eval()
    return adapted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default=str(MODELS / "voxelmorph_best.pth"))
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta",           action="store_true", help="Enable Test-Time Adaptation")
    ap.add_argument("--tta-steps",     type=int, default=30)
    ap.add_argument("--diffeomorphic", action="store_true")
    args = ap.parse_args()

    if not MANIFEST_P.exists():
        log.error("manifest_processed.csv not found.")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_P)
    manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)

    if len(manifest) == 0:
        log.error("No test subjects found in manifest.")
        sys.exit(1)

    log.info(f"Evaluating on {len(manifest)} test subjects | TTA={args.tta}")

    device = torch.device(args.device)
    model  = VoxelMorph(diffeomorphic=args.diffeomorphic).to(device)

    if not Path(args.model).exists():
        log.error(f"Model not found: {args.model}")
        sys.exit(1)

    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded model from Epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.6f})")

    mask_transformer = SpatialTransformer(size=(160, 192, 160), mode="nearest").to(device)

    rows    = []
    t_start = time.time()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Evaluating"):
        sid = row["subject_id"]
        out = DATA_PROC / sid

        mr_path      = str(out / f"{sid}_mr_norm.nii.gz")
        ct_path      = str(out / f"{sid}_ct_norm.nii.gz")
        mr_mask_path = str(out / f"{sid}_mr_brain_mask.nii.gz")
        ct_mask_path = str(out / f"{sid}_ct_mask.nii.gz")
        warped_ct_path   = str(out / f"{sid}_ct_voxelmorph.nii.gz")
        warped_mask_path = str(out / f"{sid}_ct_mask_voxelmorph.nii.gz")

        if not (Path(mr_path).exists() and Path(ct_path).exists()):
            log.warning(f"Inputs missing for {sid} — skipping.")
            continue

        try:
            mr_t    = torch.from_numpy(nib.load(mr_path).get_fdata().astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            ct_t    = torch.from_numpy(nib.load(ct_path).get_fdata().astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
            mask_t  = torch.from_numpy(nib.load(ct_mask_path).get_fdata().astype("float32")).unsqueeze(0).unsqueeze(0).to(device)

            # Test-Time Adaptation: fine-tune on this specific subject
            active_model = test_time_adapt(model, mr_t, ct_t, device,
                                           steps=args.tta_steps) if args.tta else model

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                warped_ct, dvf = active_model(mr_t, ct_t)
                warped_mask    = mask_transformer(mask_t, dvf)

            save_nifti(warped_ct,   mr_path, warped_ct_path)
            save_nifti(warped_mask, mr_path, warped_mask_path, is_mask=True)

            m = compute_all_metrics(
                mr_path, warped_ct_path, mr_mask_path, warped_mask_path,
                method="voxelmorph",
            )
            rows.append({"subject_id": sid, "split": "test", "status": "ok", **m})

        except Exception as exc:
            log.error(f"{sid}: {exc}")
            rows.append({"subject_id": sid, "split": "test", "status": f"error: {exc}"})

    results = pd.DataFrame(rows)
    ensure_dir(RESULTS)
    out_csv = RESULTS / "baseline_metrics_voxelmorph.csv"
    results.to_csv(out_csv, index=False)
    log.info(f"Saved: {out_csv}")

    ok      = results[results["status"] == "ok"]
    elapsed = time.time() - t_start

    print("\n" + "=" * 65)
    print(f"FINAL EVALUATION  |  TTA={'ON' if args.tta else 'OFF'}")
    print("=" * 65)

    for metric, label, note in [
        ("dice", "Dice (brain-mask overlap)", "target > 0.85"),
        ("hd95", "HD95 in mm",                "target < 5.0"),
        ("ncc",  "NCC",                        "higher is better"),
    ]:
        if metric not in ok.columns:
            continue
        vals = ok[metric].dropna()
        if vals.empty:
            continue
        print(f"\n  {label} ({note}):")
        print(f"    Mean ± Std : {vals.mean():.4f} ± {vals.std():.4f}")
        print(f"    Median     : {vals.median():.4f}")

    print(f"\n  Subjects OK : {len(ok)} / {len(results)}")
    print(f"  Wall time   : {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
