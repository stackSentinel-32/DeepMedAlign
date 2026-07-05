"""
visualize_difference_maps.py
-----------------------------
Generates difference-map visualisations for all registered subjects.
Each figure shows: MRI | warped CT | difference heatmap, for 3 planes.

Also saves a per-subject stats CSV (difference_map_stats_<method>.csv)
which Week 3 anomaly detection uses as its baseline noise floor.

Usage
-----
  python scripts/visualize_difference_maps.py --method affine
  python scripts/visualize_difference_maps.py --method affine --subj 1BA001
  python scripts/visualize_difference_maps.py --method bspline --n 10
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
from src.difference_maps import (
    compute_difference_map,
    normalize_diff_by_local_std,
    difference_map_stats,
)
from src.utils import get_logger, ensure_dir

log = get_logger("visualize_diff")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_arr(path: str) -> np.ndarray:
    import nibabel as nib
    return nib.load(path).get_fdata().astype("float32")


def _norm_display(arr: np.ndarray) -> np.ndarray:
    """Percentile normalisation to [0, 1] for display."""
    nz = arr[arr != 0]
    if len(nz) == 0:
        return arr
    lo, hi = np.percentile(nz, 2), np.percentile(nz, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-subject figure
# ---------------------------------------------------------------------------

def save_diff_visualization(
    mr_path:   str,
    ct_path:   str,
    mask_path: str,
    out_path:  str,
    subj_id:   str,
    method:    str = "affine",
) -> dict:
    """Save a 3×3 grid: 3 anatomical planes × (MRI | CT | diff heatmap).

    The difference heatmap is coloured with the 'hot' colormap capped at
    3 standard deviations — values above this are registration errors or
    genuine anatomical anomalies.

    Parameters
    ----------
    mr_path   : preprocessed MRI
    ct_path   : registered CT (*_ct_{method}.nii.gz)
    mask_path : brain mask (*_mr_brain_mask.nii.gz)
    out_path  : output PNG path
    subj_id   : subject identifier (for title)
    method    : registration method label

    Returns
    -------
    dict of difference map statistics
    """
    mr   = _load_arr(mr_path)
    ct   = _load_arr(ct_path)
    mask = _load_arr(mask_path) if Path(mask_path).exists() else None

    diff      = compute_difference_map(mr, ct, mask, method="absolute")
    diff_norm = normalize_diff_by_local_std(diff, mask)
    stats     = difference_map_stats(diff_norm, mask)

    d, h, w = mr.shape
    slices = [
        ("Axial",    mr[d // 2, :, :], ct[d // 2, :, :], diff_norm[d // 2, :, :]),
        ("Coronal",  mr[:, h // 2, :], ct[:, h // 2, :], diff_norm[:, h // 2, :]),
        ("Sagittal", mr[:, :, w // 2], ct[:, :, w // 2], diff_norm[:, :, w // 2]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), facecolor="#0A0A0A")
    fig.suptitle(
        f"{subj_id}  |  Difference Map  |  Method: {method.upper()}\n"
        f"Mean diff: {stats['diff_mean']:.3f}  "
        f"P95: {stats['diff_p95']:.3f}  "
        f"Max: {stats['diff_max']:.3f}  (z-score units)",
        color="white", fontsize=12, y=0.985,
    )
    col_titles = ["MRI (fixed)", "CT warped", "Difference (z-score)"]

    last_im = None
    for row, (plane, mr_sl, ct_sl, diff_sl) in enumerate(slices):
        mr_n = _norm_display(mr_sl)
        ct_n = _norm_display(ct_sl)

        axes[row, 0].imshow(mr_n.T,   cmap="gray", origin="lower", aspect="auto")
        axes[row, 1].imshow(ct_n.T,   cmap="gray", origin="lower", aspect="auto")
        last_im = axes[row, 2].imshow(
            diff_sl.T, cmap="hot", origin="lower",
            aspect="auto", vmin=0, vmax=3.0,
        )

        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, color="white",
                                         fontsize=10, pad=6)
        axes[row, 0].set_ylabel(plane, color="white", fontsize=9)

        for col in range(3):
            axes[row, col].tick_params(colors="white", labelsize=0)
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#333333")

    # Colorbar on the difference column
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[:, 2], shrink=0.7, pad=0.02)
        cbar.set_label("Std deviations from typical diff",
                       color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    log.info(f"Saved: {out_path}")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualise MRI-CT difference maps for all subjects."
    )
    ap.add_argument("--method", default="affine",
                    choices=["rigid", "affine", "bspline"],
                    help="Registration method to evaluate.")
    ap.add_argument("--subj",  default=None,
                    help="Process a single subject by ID.")
    ap.add_argument("--n",     type=int, default=None,
                    help="Process only the first N subjects.")
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

    out_dir = RESULTS / "figures" / "difference_maps" / args.method
    ensure_dir(out_dir)

    all_stats = []
    saved     = 0

    for _, row in manifest.iterrows():
        sid       = row["subject_id"]
        out       = DATA_PROC / sid
        mr_path   = str(out / f"{sid}_mr_norm.nii.gz")
        ct_path   = str(out / f"{sid}_ct_{args.method}.nii.gz")
        mask_path = str(out / f"{sid}_mr_brain_mask.nii.gz")

        if not Path(mr_path).exists() or not Path(ct_path).exists():
            log.warning(f"{sid}: preprocessed or registered file missing — skipping.")
            continue

        out_path = str(out_dir / f"{sid}_{args.method}_diffmap.png")
        try:
            stats = save_diff_visualization(
                mr_path, ct_path, mask_path,
                out_path, sid, method=args.method,
            )
            stats["subject_id"] = sid
            stats["split"]      = row.get("split", "")
            all_stats.append(stats)
            saved += 1
        except Exception as exc:
            log.error(f"{sid}: {exc}")

    # Save per-subject stats CSV
    if all_stats:
        stats_df  = pd.DataFrame(all_stats)
        stats_csv = RESULTS / f"difference_map_stats_{args.method}.csv"
        stats_df.to_csv(stats_csv, index=False)
        log.info(f"Stats saved: {stats_csv}")

        print(f"\n{'=' * 55}")
        print(f"Difference map stats  |  Method: {args.method}")
        print(f"  diff_mean : {stats_df['diff_mean'].mean():.4f}")
        print(f"  diff_p95  : {stats_df['diff_p95'].mean():.4f}")
        print(f"{'=' * 55}")

    print(f"\nSaved {saved} difference-map PNGs to: {out_dir}")


if __name__ == "__main__":
    main()
