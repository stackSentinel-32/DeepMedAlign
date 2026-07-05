"""
checkerboard_qc.py
------------------
Generates checkerboard overlay PNGs for visual registration QC.
Supports comparing across rigid, affine, and bspline in one figure.

A checkerboard alternates tiles between the fixed (MRI) and warped
moving (CT) image.  If registration is good, anatomical structures
(ventricles, brain outline) are continuous across tile boundaries.
If registration failed, you see jagged jumps at tile edges.

Usage
-----
  python scripts/checkerboard_qc.py --method affine
  python scripts/checkerboard_qc.py --method affine --subj 1BA001
  python scripts/checkerboard_qc.py --compare-all --subj 1BA001
  python scripts/checkerboard_qc.py --method affine --n 5
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW, DATA_PROC, RESULTS
from src.utils import get_logger, ensure_dir

log = get_logger("checkerboard_qc")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_arr(path: str) -> np.ndarray:
    """Load a NIfTI file and return a float32 numpy array."""
    import nibabel as nib
    return nib.load(path).get_fdata().astype("float32")


def norm_display(arr: np.ndarray) -> np.ndarray:
    """Percentile-based normalisation to [0, 1] for consistent display.

    Uses the 2nd and 98th percentiles of non-zero voxels so that bright
    outliers do not wash out the contrast.
    """
    nz = arr[arr != 0]
    if len(nz) == 0:
        return arr
    lo, hi = np.percentile(nz, 2), np.percentile(nz, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def checkerboard_2d(img_a: np.ndarray,
                    img_b: np.ndarray,
                    tile:  int = 20) -> np.ndarray:
    """Interleave two 2-D images in a checkerboard pattern.

    Tiles at (row_block + col_block) even positions show img_a (MRI);
    odd positions show img_b (CT).  The result has the same shape as
    the inputs.

    Parameters
    ----------
    img_a, img_b : 2-D float arrays of identical shape
    tile         : tile edge length in pixels

    Returns
    -------
    2-D float array with values from img_a and img_b interleaved
    """
    if img_a.shape != img_b.shape:
        raise ValueError(
            f"checkerboard_2d: shape mismatch {img_a.shape} vs {img_b.shape}"
        )
    h, w   = img_a.shape
    result = np.zeros_like(img_a)
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            src = img_a if (i // tile + j // tile) % 2 == 0 else img_b
            result[i : i + tile, j : j + tile] = src[i : i + tile, j : j + tile]
    return result


# ---------------------------------------------------------------------------
# Single-method figure: MRI | warped CT | checkerboard — 3 planes
# ---------------------------------------------------------------------------

def save_single_method_checkerboard(
    mr_path:   str,
    ct_warped: str,
    out_path:  str,
    subj_id:   str,
    method:    str = "affine",
    tile_size: int = 20,
) -> None:
    """Save a 3×3 grid (3 planes × MRI / CT / checkerboard) as a PNG.

    Parameters
    ----------
    mr_path   : path to preprocessed MRI   (*_mr_norm.nii.gz)
    ct_warped : path to registered CT      (*_ct_{method}.nii.gz)
    out_path  : output PNG path
    subj_id   : subject identifier (for title)
    method    : registration method label
    tile_size : checkerboard tile edge length (pixels)
    """
    mr = load_arr(mr_path)
    ct = load_arr(ct_warped)

    d, h, w = mr.shape
    slices = [
        ("Axial",    mr[d // 2, :, :], ct[d // 2, :, :]),
        ("Coronal",  mr[:, h // 2, :], ct[:, h // 2, :]),
        ("Sagittal", mr[:, :, w // 2], ct[:, :, w // 2]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14), facecolor="#0A0A0A")
    fig.suptitle(
        f"{subj_id}  |  Registration QC  |  Method: {method.upper()}\n"
        "Left: MRI (fixed)   Centre: CT (warped)   Right: Checkerboard",
        color="white", fontsize=12, y=0.985,
    )
    col_titles = ["MRI (fixed)", "CT warped", "Checkerboard"]

    for row, (plane, mr_sl, ct_sl) in enumerate(slices):
        mr_n = norm_display(mr_sl)
        ct_n = norm_display(ct_sl)
        cb   = checkerboard_2d(mr_n, ct_n, tile=tile_size)

        for col, img in enumerate([mr_n, ct_n, cb]):
            ax = axes[row, col]
            ax.imshow(img.T, cmap="gray", origin="lower",
                      aspect="auto", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(col_titles[col], color="white",
                             fontsize=10, pad=6)
            ax.set_ylabel(plane, color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=0)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Multi-method comparison figure: rigid vs affine vs bspline
# ---------------------------------------------------------------------------

def save_compare_all_methods(
    mr_path:       str,
    out_dir:       str,
    subj_id:       str,
    data_proc_dir: Path,
) -> None:
    """Compare rigid / affine / bspline side-by-side on one axial slice.

    Shows how registration quality improves stage by stage.  Checkerboard
    tiles from all available methods are placed in a single row.

    Parameters
    ----------
    mr_path       : path to preprocessed MRI
    out_dir       : output directory for the comparison PNG
    subj_id       : subject identifier
    data_proc_dir : data/processed/<subj_id>/ directory
    """
    mr    = load_arr(mr_path)
    d     = mr.shape[0]
    mr_sl = norm_display(mr[d // 2, :, :])

    methods   = ["rigid", "affine", "bspline"]
    available = [
        (m, data_proc_dir / f"{subj_id}_ct_{m}.nii.gz")
        for m in methods
        if (data_proc_dir / f"{subj_id}_ct_{m}.nii.gz").exists()
    ]

    if not available:
        log.error(f"{subj_id}: no registered CT outputs found for any method.")
        return

    n    = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), facecolor="#0A0A0A")
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"{subj_id}  |  Registration progression: "
        + " -> ".join(m for m, _ in available),
        color="white", fontsize=13,
    )

    for ax, (method, ct_path) in zip(axes, available):
        ct    = load_arr(str(ct_path))
        ct_sl = norm_display(ct[d // 2, :, :])
        cb    = checkerboard_2d(mr_sl, ct_sl, tile=20)
        ax.imshow(cb.T, cmap="gray", origin="lower", aspect="auto",
                  vmin=0, vmax=1)
        ax.set_title(method.upper(), color="white", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    out_path = str(Path(out_dir) / f"{subj_id}_method_comparison.png")
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    log.info(f"Saved comparison: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Checkerboard QC visualisation for MRI-CT registration."
    )
    ap.add_argument("--method", default="affine",
                    choices=["rigid", "affine", "bspline"],
                    help="Registration method to visualise.")
    ap.add_argument("--subj",  default=None,
                    help="Process a single subject by ID.")
    ap.add_argument("--n",     type=int, default=None,
                    help="Process only the first N subjects.")
    ap.add_argument("--tile",  type=int, default=20,
                    help="Checkerboard tile size in pixels (default: 20).")
    ap.add_argument("--compare-all", action="store_true",
                    help="Side-by-side comparison of all methods (requires --subj).")
    args = ap.parse_args()

    manifest_path = DATA_RAW / "manifest.csv"
    if not manifest_path.exists():
        log.error("manifest.csv not found. Run build_manifest.py first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    if args.subj:
        manifest = manifest[manifest["subject_id"] == args.subj]
    if args.n:
        manifest = manifest.head(args.n)
    if manifest.empty:
        log.error("No subjects matched filter.")
        sys.exit(1)

    # ── Compare-all mode ────────────────────────────────────────────────────
    if args.compare_all:
        if not args.subj:
            log.error("--compare-all requires --subj <id>.")
            sys.exit(1)
        sid     = args.subj
        out     = DATA_PROC / sid
        mr_path = str(out / f"{sid}_mr_norm.nii.gz")
        if not Path(mr_path).exists():
            log.error(f"{sid}: MR not preprocessed yet.")
            sys.exit(1)
        out_dir = RESULTS / "figures" / "checkerboard" / "comparison"
        save_compare_all_methods(mr_path, str(out_dir), sid, out)
        return

    # ── Single-method mode ──────────────────────────────────────────────────
    out_dir = RESULTS / "figures" / "checkerboard" / args.method
    ensure_dir(out_dir)
    saved, skipped = 0, 0

    for _, row in manifest.iterrows():
        sid     = row["subject_id"]
        out     = DATA_PROC / sid
        mr_path = str(out / f"{sid}_mr_norm.nii.gz")
        ct_path = str(out / f"{sid}_ct_{args.method}.nii.gz")

        if not Path(mr_path).exists():
            log.warning(f"{sid}: MR not preprocessed — skipping.")
            skipped += 1
            continue
        if not Path(ct_path).exists():
            log.warning(f"{sid}: {args.method} registration missing — skipping.")
            skipped += 1
            continue

        out_path = str(out_dir / f"{sid}_{args.method}_checkerboard.png")
        try:
            save_single_method_checkerboard(
                mr_path, ct_path, out_path, sid,
                method=args.method, tile_size=args.tile,
            )
            saved += 1
        except Exception as exc:
            log.error(f"{sid}: {exc}")
            skipped += 1

    print(f"\n{'=' * 55}")
    print(f"Checkerboard QC  |  Method: {args.method}")
    print(f"  Saved   : {saved}")
    print(f"  Skipped : {skipped}")
    print(f"  Output  : {out_dir}")
    print(f"{'=' * 55}")
    print("\nWhat to look for:")
    print("  GOOD: tissue boundaries continuous across tile edges")
    print("  BAD : jagged jumps, brain in wrong position")


if __name__ == "__main__":
    main()
