"""
qc_dashboard.py
---------------
Single-page registration quality dashboard.
Combines baseline metrics (Dice, HD95) and difference-map stats
across all methods into one visual report.

Run:
  python scripts/qc_dashboard.py
  python scripts/qc_dashboard.py --out results/figures/my_dashboard.png
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
from src.config import RESULTS
from src.utils import get_logger, ensure_dir

log = get_logger("qc_dashboard")

METHODS = ["rigid", "affine", "bspline"]
COLORS  = {"rigid": "#A32D2D", "affine": "#1A7DBF", "bspline": "#1D8F75"}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_metrics(method: str) -> pd.DataFrame:
    """Load baseline_metrics_<method>.csv and return only OK rows."""
    p = RESULTS / f"baseline_metrics_{method}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["status"] == "ok"] if "status" in df.columns else df


def _load_diff_stats(method: str) -> pd.DataFrame:
    """Load difference_map_stats_<method>.csv."""
    p = RESULTS / f"difference_map_stats_{method}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def build_dashboard(out_path: str) -> None:
    """Build and save the 2×2 dashboard PNG.

    Panel layout:
      [0,0] Dice boxplot by method
      [0,1] HD95 boxplot by method
      [1,0] Difference-map mean scatter (per subject, per method)
      [1,1] Summary text table
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0A0A0A")
    fig.suptitle(
        "DeepMedAlign — Registration Quality Dashboard",
        color="white", fontsize=16, y=0.98,
    )

    # ── Panel [0,0]: Dice boxplot ──────────────────────────────────────────
    ax = axes[0, 0]
    dice_data, dice_labels = [], []
    for m in METHODS:
        df   = _load_metrics(m)
        if not df.empty and "dice" in df.columns:
            vals = df["dice"].dropna()
            if len(vals) > 0:
                dice_data.append(vals.values)
                dice_labels.append(m.capitalize())

    ax.set_facecolor("#111111")
    if dice_data:
        bp = ax.boxplot(dice_data, labels=dice_labels, patch_artist=True)
        for patch, label in zip(bp["boxes"], dice_labels):
            patch.set_facecolor(COLORS.get(label.lower(), "#888888"))
            patch.set_alpha(0.75)
        ax.axhline(0.85, color="yellow", linestyle="--",
                   alpha=0.6, linewidth=1.2, label="Target 0.85")
        ax.legend(facecolor="#222222", labelcolor="white", fontsize=9)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No Dice data yet\n(run compute_baseline_metrics.py)",
                color="#888888", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title("Dice Score by Method", color="white", fontsize=12)
    ax.set_ylabel("Dice", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # ── Panel [0,1]: HD95 boxplot ──────────────────────────────────────────
    ax = axes[0, 1]
    hd_data, hd_labels = [], []
    for m in METHODS:
        df = _load_metrics(m)
        if not df.empty and "hd95" in df.columns:
            vals = df["hd95"].dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                hd_data.append(vals.values)
                hd_labels.append(m.capitalize())

    ax.set_facecolor("#111111")
    if hd_data:
        bp = ax.boxplot(hd_data, labels=hd_labels, patch_artist=True)
        for patch, label in zip(bp["boxes"], hd_labels):
            patch.set_facecolor(COLORS.get(label.lower(), "#888888"))
            patch.set_alpha(0.75)
        ax.axhline(5.0, color="yellow", linestyle="--",
                   alpha=0.6, linewidth=1.2, label="Target 5 mm")
        ax.legend(facecolor="#222222", labelcolor="white", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No HD95 data yet\n(run compute_baseline_metrics.py)",
                color="#888888", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title("HD95 (mm) by Method", color="white", fontsize=12)
    ax.set_ylabel("HD95 (mm)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # ── Panel [1,0]: Difference-map mean scatter ───────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor("#111111")
    any_diff = False
    for m in METHODS:
        df = _load_diff_stats(m)
        if not df.empty and "diff_mean" in df.columns:
            ax.scatter(
                range(len(df)), df["diff_mean"],
                label=m.capitalize(),
                color=COLORS.get(m, "#888888"),
                alpha=0.65, s=18,
            )
            any_diff = True
    if not any_diff:
        ax.text(0.5, 0.5,
                "No diff stats yet\n(run visualize_difference_maps.py)",
                color="#888888", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title("Difference Map Mean (per subject)", color="white", fontsize=12)
    ax.set_xlabel("Subject index", color="white")
    ax.set_ylabel("Mean diff (z-score)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222222", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # ── Panel [1,1]: Summary text ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_facecolor("#111111")

    lines = ["SUMMARY", ""]
    for m in METHODS:
        df = _load_metrics(m)
        if df.empty:
            lines.append(f"{m.upper()}: no data")
            lines.append("")
            continue
        n          = len(df)
        dice_mean  = df["dice"].mean()  if "dice" in df.columns  else None
        hd95_mean  = df["hd95"].mean()  if "hd95" in df.columns  else None
        lines.append(f"{m.upper()}  (n={n})")
        if dice_mean is not None:
            lines.append(f"  Dice : {dice_mean:.4f}")
        if hd95_mean is not None:
            lines.append(f"  HD95 : {hd95_mean:.2f} mm")
        lines.append("")

    lines += [
        "VoxelMorph (Week 3) target:",
        "  Beat affine Dice (> 0.85)",
        "  Beat affine HD95 (< 5 mm)",
        "  Jacobian neg% < 1%",
    ]

    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, color="white",
            fontsize=11, va="top", family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0A0A0A")
    plt.close()
    log.info(f"Dashboard saved: {out_path}")
    print(f"\nDashboard saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the registration quality dashboard."
    )
    ap.add_argument(
        "--out", default=None,
        help="Output PNG path (default: results/figures/qc_dashboard.png).",
    )
    args = ap.parse_args()

    out_path = args.out or str(RESULTS / "figures" / "qc_dashboard.png")
    build_dashboard(out_path)


if __name__ == "__main__":
    main()
