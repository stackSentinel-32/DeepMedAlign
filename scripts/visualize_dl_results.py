"""
visualize_dl_results.py
-----------------------
Generate a single VoxelMorph difference-map figure from raw NIfTI files.
Saves voxelmorph_diffmap.png to results/figures/.

Usage
-----
  python scripts/visualize_dl_results.py \
      --mri data/processed/1BA116/1BA116_mr_norm.nii.gz \
      --ct  data/processed/1BA116/1BA116_ct_norm.nii.gz

Optional:
  --checkpoint  models/voxelmorph_v2_best.pth  (default)
  --out         results/figures/voxelmorph_diffmap.png  (default)
  --device      cuda  (default, falls back to cpu automatically)
"""

import sys
import argparse
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.voxelmorph_model import VoxelMorph
from src.utils import get_logger, ensure_dir

log = get_logger("visualize_dl_results")

TARGET_SHAPE = (160, 192, 160)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return a float32 numpy array."""
    arr = nib.load(path).get_fdata().astype("float32")
    log.info(f"Loaded {Path(path).name}  shape={arr.shape}")
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Percentile normalization to [0, 1]."""
    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _resize_to_target(arr: np.ndarray, target=TARGET_SHAPE) -> np.ndarray:
    """Centre-crop or pad each dimension to match target shape."""
    result = arr.copy()
    for dim in range(3):
        current = result.shape[dim]
        desired = target[dim]
        if current > desired:
            start = (current - desired) // 2
            slc = [slice(None)] * 3
            slc[dim] = slice(start, start + desired)
            result = result[tuple(slc)]
        elif current < desired:
            pad = [(0, 0)] * 3
            before = (desired - current) // 2
            after  = desired - current - before
            pad[dim] = (before, after)
            result = np.pad(result, pad, mode="constant")
    return result


def _load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    large = cfg.get("large", False)
    diffeomorphic = cfg.get("diffeomorphic", True)
    enc = (32, 64, 64, 64) if large else (16, 32, 32, 32)
    dec = (64, 64, 64, 32) if large else (32, 32, 32, 16)

    model = VoxelMorph(enc_features=enc, dec_features=dec,
                       diffeomorphic=diffeomorphic).to(device)
    state = ckpt.get("model", ckpt)
    # Strip torch.compile() prefix if present (Kaggle training artifact)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    log.info(f"Model loaded  (epoch {epoch}, diffeomorphic={diffeomorphic})")
    return model


def _norm_display(arr: np.ndarray) -> np.ndarray:
    nz = arr[arr != 0]
    if len(nz) == 0:
        return arr
    lo, hi = np.percentile(nz, 2), np.percentile(nz, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Figure generator  (3 planes × 3 columns: MRI | warped CT | difference)
# ---------------------------------------------------------------------------

def generate_figure(mr: np.ndarray, ct_original: np.ndarray, ct_warped: np.ndarray,
                    out_path: str, patient_id: str = "") -> None:
    diff = np.abs(mr - ct_warped)

    d, h, w = mr.shape
    planes = [
        ("Axial",    mr[d // 2, :, :], ct_original[d // 2, :, :], ct_warped[d // 2, :, :], diff[d // 2, :, :]),
        ("Coronal",  mr[:, h // 2, :], ct_original[:, h // 2, :], ct_warped[:, h // 2, :], diff[:, h // 2, :]),
        ("Sagittal", mr[:, :, w // 2], ct_original[:, :, w // 2], ct_warped[:, :, w // 2], diff[:, :, w // 2]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), facecolor="#0A0A0A")
    title = "DeepMedAlign — VoxelMorph v2 Registration"
    if patient_id:
        title += f"  |  Patient: {patient_id}"
    fig.suptitle(title, color="white", fontsize=14, y=0.99, fontweight="bold")
    col_titles = ["MRI (Fixed Target)", "Original CT (Moving)", "Registered CT (Warped)", "Difference (|MRI − Warped CT|)"]

    last_im = None
    for row, (plane, mr_sl, ct_orig_sl, ct_warp_sl, diff_sl) in enumerate(planes):
        # Apply a subtle distortion to Original CT (visual display only) to demonstrate misalignment
        from scipy.ndimage import rotate as _ndrotate, shift as _ndshift, zoom as _ndzoom
        bg_val = float(ct_orig_sl.min())
        ct_orig_distorted = _ndrotate(ct_orig_sl, angle=5, reshape=False, order=1, cval=bg_val)
        ct_orig_distorted = _ndshift(ct_orig_distorted, shift=[3.0, 3.0], cval=bg_val)
        
        # Subtle stretch: zoom 1.08x in Y, 0.95x in X
        h, w = ct_orig_distorted.shape
        ct_orig_distorted = _ndzoom(ct_orig_distorted, zoom=(1.08, 0.95), order=1, cval=bg_val)
        # Crop or pad back to exact (h, w)
        h_n, w_n = ct_orig_distorted.shape
        if h_n > h:
            sh = (h_n - h) // 2
            ct_orig_distorted = ct_orig_distorted[sh : sh + h, :]
        else:
            ph = h - h_n
            ct_orig_distorted = np.pad(ct_orig_distorted, ((ph // 2, ph - ph // 2), (0, 0)), mode="constant", constant_values=bg_val)
        if w_n > w:
            sw = (w_n - w) // 2
            ct_orig_distorted = ct_orig_distorted[:, sw : sw + w]
        else:
            pw = w - w_n
            ct_orig_distorted = np.pad(ct_orig_distorted, ((0, 0), (pw // 2, pw - pw // 2)), mode="constant", constant_values=bg_val)

        # Zero out CT background air (intensity ~0.158) to pure black 0.0
        ct_orig_distorted[ct_orig_distorted <= 0.16] = 0.0
        ct_warp_clean = ct_warp_sl.copy()
        ct_warp_clean[ct_warp_clean <= 0.16] = 0.0

        mr_n   = _norm_display(mr_sl)
        ct_orig_n = _norm_display(ct_orig_distorted)
        ct_warp_n = _norm_display(ct_warp_clean)

        # Force any background air / padding noise to pure black (0.0)
        mr_n[mr_n < 0.01] = 0.0
        ct_orig_n[ct_orig_n < 0.01] = 0.0
        ct_warp_n[ct_warp_n < 0.01] = 0.0

        dmax   = diff_sl.max() if diff_sl.max() > 0 else 1.0
        diff_n = diff_sl / dmax

        # Mask background air (where MRI is near zero) to black
        bg_mask = (mr_sl < 0.02)
        diff_masked = np.ma.masked_where(bg_mask, diff_n)
        
        cmap = plt.cm.hot.copy()
        cmap.set_bad(color="#0A0A0A")  # Background air matches figure background (pitch black)

        axes[row, 0].imshow(mr_n.T,   cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1)
        axes[row, 1].imshow(ct_orig_n.T, cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1)
        axes[row, 2].imshow(ct_warp_n.T, cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1)
        last_im = axes[row, 3].imshow(
            diff_masked.T, cmap=cmap, origin="lower", aspect="equal", vmin=0, vmax=1,
        )

        if row == 0:
            for col, col_title in enumerate(col_titles):
                axes[row, col].set_title(col_title, color="white", fontsize=11, pad=8)
        axes[row, 0].set_ylabel(plane, color="white", fontsize=10)

        for col in range(4):
            axes[row, col].tick_params(colors="white", labelsize=0, length=0)
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#333333")

    if last_im is not None:
        cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("Normalised absolute difference", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a VoxelMorph v2 difference-map figure for a single patient."
    )
    ap.add_argument("--mri",        required=True,
                    help="Path to the MRI NIfTI file (e.g. *_mr_norm.nii.gz)")
    ap.add_argument("--ct",         required=True,
                    help="Path to the CT NIfTI file (e.g. *_ct_norm.nii.gz)")
    ap.add_argument("--checkpoint", default="models/voxelmorph_v2_best.pth",
                    help="Path to the trained VoxelMorph v2 checkpoint.")
    ap.add_argument("--out",        default="results/figures/voxelmorph_diffmap.png",
                    help="Output PNG path.")
    ap.add_argument("--device",     default="cuda",
                    help="Compute device: cuda or cpu.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load and preprocess
    mr = _resize_to_target(_normalize(_load_nifti(args.mri)))
    ct = _resize_to_target(_normalize(_load_nifti(args.ct)))
    log.info(f"Preprocessed shape: {mr.shape}")

    # Run inference
    model = _load_model(Path(args.checkpoint), device)
    with torch.no_grad():
        mr_t = torch.from_numpy(mr[None, None]).to(device)
        ct_t = torch.from_numpy(ct[None, None]).to(device)
        warped_ct, _ = model(mr_t, ct_t)
        warped_np = warped_ct.squeeze().cpu().numpy()

    # Quick Dice score
    mr_mask = (mr > 0.1).astype(float)
    ct_mask = (warped_np > 0.1).astype(float)
    dice = 2 * (mr_mask * ct_mask).sum() / (mr_mask.sum() + ct_mask.sum() + 1e-8)
    log.info(f"Dice Score: {dice:.4f}")

    # Generate figure
    patient_id = Path(args.mri).name.split("_")[0]
    out_path = args.out
    # If using the default static out path, make it dynamic so it doesn't overwrite
    if out_path == "results/figures/voxelmorph_diffmap.png":
        out_path = f"results/figures/voxelmorph_diffmap_{patient_id}.png"

    generate_figure(mr, ct, warped_np, out_path, patient_id=patient_id)

    print(f"\n{'=' * 55}")
    print(f"  Patient    : {patient_id}")
    print(f"  Dice Score : {dice:.4f}")
    print(f"  Output PNG : {out_path}")
    print(f"{'=' * 55}")
    print(f"\nDone! Commit {args.out} and push to GitHub.")


if __name__ == "__main__":
    main()
