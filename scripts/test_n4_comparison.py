"""
test_n4_comparison.py
---------------------
Runs N4 bias correction on a synthetic noisy volume and saves
a before/after comparison PNG.

Run:
    python scripts/test_n4_comparison.py
"""

import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess_mri import n4_bias_correction
from src.config import RESULTS
from src.utils import get_logger, ensure_dir

log = get_logger("test_n4")


def make_biased_phantom():
    """Create a synthetic MRI-like volume with bias field."""
    size = 64
    arr  = np.zeros((size, size, size), dtype="float32")

    # Brain-like sphere
    cx, cy, cz = size // 2, size // 2, size // 2
    for z in range(size):
        for y in range(size):
            for x in range(size):
                if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < (size // 3)**2:
                    arr[z, y, x] = 800.0 + np.random.randn() * 50

    # Add synthetic bias field (gradient across x-axis)
    bias = np.linspace(0.5, 1.5, size)
    arr  = arr * bias[np.newaxis, np.newaxis, :]

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def main():
    ensure_dir(RESULTS / "figures" / "n4_comparison")

    log.info("Creating synthetic biased phantom...")
    img_before = make_biased_phantom()

    log.info("Running N4 bias correction...")
    img_after = n4_bias_correction(img_before)

    arr_b = sitk.GetArrayFromImage(img_before)
    arr_a = sitk.GetArrayFromImage(img_after)
    mid   = arr_b.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#111")
    fig.suptitle("N4 Bias Field Correction — Before vs After",
                 color="white", fontsize=13)

    for col, label in enumerate(["Axial", "Coronal", "Sagittal"]):
        if label == "Axial":
            sl_b = arr_b[mid, :, :]
            sl_a = arr_a[mid, :, :]
        elif label == "Coronal":
            sl_b = arr_b[:, mid, :]
            sl_a = arr_a[:, mid, :]
        else:
            sl_b = arr_b[:, :, mid]
            sl_a = arr_a[:, :, mid]

        vmax = max(sl_b.max(), sl_a.max())

        axes[0, col].imshow(sl_b.T, cmap="gray",
                            origin="lower", vmin=0, vmax=vmax)
        axes[0, col].set_title(f"Before N4 — {label}",
                               color="white", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(sl_a.T, cmap="gray",
                            origin="lower", vmin=0, vmax=vmax)
        axes[1, col].set_title(f"After N4 — {label}",
                               color="white", fontsize=10)
        axes[1, col].axis("off")

    stats = (f"Before: mean={arr_b[arr_b > 0].mean():.1f}  "
             f"std={arr_b[arr_b > 0].std():.1f}\n"
             f"After:  mean={arr_a[arr_a > 0].mean():.1f}  "
             f"std={arr_a[arr_a > 0].std():.1f}")
    fig.text(0.5, 0.01, stats, ha="center",
             color="#888", fontsize=9, family="monospace")

    out = RESULTS / "figures" / "n4_comparison" / "synthetic_n4_test.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    log.info(f"Saved: {out}")
    print(f"\nOpen this file to verify N4 worked: {out}")


if __name__ == "__main__":
    main()
