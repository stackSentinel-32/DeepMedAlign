"""
load_and_view.py
----------------
Loads one MRI + CT pair and saves a 6-panel PNG showing
axial, coronal, sagittal views for both modalities side by side.

Run:
    python scripts/load_and_view.py
    python scripts/load_and_view.py --subj 1PA001
    python scripts/load_and_view.py --subj 1PA000 --out results/figures/raw_views/
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SYNTHRAD, RESULTS
from src.utils import get_logger, ensure_dir

log = get_logger("load_and_view")


def load_volume(path: str):
    """Load NIfTI volume. Returns (array float32, spacing tuple)."""
    import nibabel as nib
    img  = nib.load(path)
    arr  = img.get_fdata().astype("float32")
    zoom = img.header.get_zooms()
    return arr, tuple(round(float(z), 3) for z in zoom), img.shape


def get_mid_slices(vol: np.ndarray):
    """Return middle axial, coronal, sagittal slices."""
    d, h, w = vol.shape[:3]
    axial    = vol[d // 2, :, :]
    coronal  = vol[:, h // 2, :]
    sagittal = vol[:, :, w // 2]
    return axial, coronal, sagittal


def percentile_window(arr: np.ndarray, lo=2, hi=98):
    """Clip and normalise array to [0,1] using percentiles."""
    lo_val = np.percentile(arr[arr != 0], lo) if (arr != 0).any() else arr.min()
    hi_val = np.percentile(arr[arr != 0], hi) if (arr != 0).any() else arr.max()
    clipped = np.clip(arr, lo_val, hi_val)
    denom   = hi_val - lo_val + 1e-8
    return (clipped - lo_val) / denom


def save_pair_view(mr_path: str,
                   ct_path: str,
                   out_path: str,
                   subj_id: str):
    """
    Save a 2x3 grid:
      Row 0 = MR  (axial | coronal | sagittal)
      Row 1 = CT  (axial | coronal | sagittal)
    """
    log.info(f"Loading {subj_id}...")

    # Load
    try:
        mr_arr, mr_sp, mr_sh = load_volume(mr_path)
        ct_arr, ct_sp, ct_sh = load_volume(ct_path)
    except Exception as e:
        log.error(f"Could not load {subj_id}: {e}")
        log.warning("Dummy subjects (empty files) cannot be loaded — "
                    "this is expected until real SynthRAD data arrives.")
        return

    # Log key stats
    log.info(f"  MR shape={mr_sh}  spacing={mr_sp}")
    log.info(f"  CT shape={ct_sh}  spacing={ct_sp}")
    log.info(f"  MR range: [{mr_arr.min():.1f}, {mr_arr.max():.1f}]")
    log.info(f"  CT range: [{ct_arr.min():.1f}, {ct_arr.max():.1f}] HU")

    # Slices
    mr_slices = get_mid_slices(mr_arr)
    ct_slices = get_mid_slices(ct_arr)
    labels    = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    # Plot
    fig = plt.figure(figsize=(16, 10), facecolor="#111111")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.08, wspace=0.04)

    for col, (label, mr_sl, ct_sl) in enumerate(
            zip(labels, mr_slices, ct_slices)):

        # MR row
        ax_mr = fig.add_subplot(gs[0, col])
        ax_mr.imshow(percentile_window(mr_sl).T,
                     cmap="gray", origin="lower", aspect="auto")
        ax_mr.set_title(f"MR — {label}", color="white",
                        fontsize=11, pad=6)
        ax_mr.axis("off")

        # CT row
        ax_ct = fig.add_subplot(gs[1, col])
        ct_display = np.clip(ct_sl, -200, 800)
        ct_norm    = (ct_display - ct_display.min()) / \
                     (ct_display.max() - ct_display.min() + 1e-8)
        ax_ct.imshow(ct_norm.T, cmap="gray",
                     origin="lower", aspect="auto")
        ax_ct.set_title(f"CT — {label} (HU −200→800)",
                        color="white", fontsize=11, pad=6)
        ax_ct.axis("off")

    fig.suptitle(
        f"Subject: {subj_id}  |  MR (top) vs CT (bottom)  |  RAW unprocessed",
        color="white", fontsize=13, y=0.98
    )

    # Stats text
    stats = (
        f"MR: shape={mr_sh}  spacing={mr_sp}mm  "
        f"range=[{mr_arr.min():.0f}, {mr_arr.max():.0f}]\n"
        f"CT: shape={ct_sh}  spacing={ct_sp}mm  "
        f"range=[{ct_arr.min():.0f}, {ct_arr.max():.0f}] HU"
    )
    fig.text(0.5, 0.01, stats, ha="center", color="#888888",
             fontsize=8, family="monospace")

    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#111111")
    plt.close()
    log.info(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", default=None,
                    help="Subject ID (e.g. 1PA000). Default: first found.")
    ap.add_argument("--out", default=None,
                    help="Output directory. Default: results/figures/raw_views/")
    args = ap.parse_args()

    # Find subject
    if args.subj:
        subj_dir = SYNTHRAD / args.subj
    else:
        candidates = [d for d in sorted(SYNTHRAD.iterdir())
                      if d.is_dir()]
        if not candidates:
            log.error(f"No subjects found in {SYNTHRAD}")
            sys.exit(1)
        subj_dir = candidates[0]

    subj_id = subj_dir.name
    out_dir = Path(args.out) if args.out else RESULTS / "figures" / "raw_views"
    out_path = str(out_dir / f"{subj_id}_raw_view.png")

    save_pair_view(
        str(subj_dir / "mr.nii.gz"),
        str(subj_dir / "ct.nii.gz"),
        out_path,
        subj_id,
    )


if __name__ == "__main__":
    main()
