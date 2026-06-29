"""
metrics.py
----------
Registration evaluation metrics used to establish the classical baseline.

These numbers are the FLOOR that VoxelMorph must beat in Week 3.

Metrics
-------
Dice  -- brain-mask overlap          (higher is better;  target > 0.85)
HD95  -- 95th-percentile Hausdorff   (lower  is better;  target < 5 mm)
NCC   -- normalised cross-correlation (secondary/sanity check)
Jac   -- Jacobian determinant stats   (neg% must be 0 for classical reg)

Important notes
---------------
- Dice and HD95 both require a warped CT mask in MRI space.
- HD95 is computed via distance transforms (efficient on 3-D volumes).
- NCC between MRI and CT is NOT meaningful due to the non-linear
  intensity relationship; use it only as a secondary/sanity check.
- Jacobian determinant < 0 means physically impossible folding.
  Classical registration should always yield neg% == 0.
  VoxelMorph: target neg% < 1 %; use diffeomorphic mode if exceeded.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Dice Similarity Coefficient
# ---------------------------------------------------------------------------

def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

    Formula: 2 * |A ∩ B| / (|A| + |B|)
    Range:   [0, 1].  1 = perfect overlap,  0 = no overlap.

    Brain-mask registration guidelines:
      > 0.90 = excellent
      > 0.85 = good     (target for affine)
      > 0.75 = acceptable
      < 0.70 = registration likely failed
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = (a & b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0  # both masks empty → trivially identical
    return float(2.0 * intersection / denom)


# ---------------------------------------------------------------------------
# Hausdorff Distance (95th percentile)
# ---------------------------------------------------------------------------

def hausdorff95(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    voxel_size: float = 1.0,
) -> float:
    """95th-percentile Hausdorff distance between two binary masks (in mm).

    Measures the maximum surface-to-surface distance after removing the
    top 5 % of outliers (more robust than the full Hausdorff).
    Computed via Euclidean distance transforms.

    Brain-mask registration guidelines:
      <  3 mm = excellent
      <  5 mm = good
      <  8 mm = acceptable
      > 10 mm = registration likely failed
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    if a.sum() == 0 or b.sum() == 0:
        return float("inf")

    dist_to_b = distance_transform_edt(~b) * voxel_size
    dist_to_a = distance_transform_edt(~a) * voxel_size

    h_a_to_b = np.percentile(dist_to_b[a], 95)
    h_b_to_a = np.percentile(dist_to_a[b], 95)

    return float(max(h_a_to_b, h_b_to_a))


# ---------------------------------------------------------------------------
# Normalised Cross-Correlation
# ---------------------------------------------------------------------------

def normalised_cross_correlation(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask:  np.ndarray = None,
) -> float:
    """Normalised cross-correlation between two intensity volumes.

    Range: [-1, 1].  1 = identical,  0 = uncorrelated,  -1 = inverse.

    WARNING: NCC is only meaningful for same-modality comparisons.
    MRI and CT do not share a linear intensity relationship.
    Use as a secondary / sanity-check metric only.
    Primary metrics are Dice and HD95.
    """
    if mask is not None:
        a = img_a[mask.astype(bool)].ravel()
        b = img_b[mask.astype(bool)].ravel()
    else:
        a = img_a.ravel()
        b = img_b.ravel()

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-8
    return float((a * b).sum() / denom)


# ---------------------------------------------------------------------------
# Jacobian determinant statistics
# ---------------------------------------------------------------------------

def jacobian_stats(flow: np.ndarray) -> dict:
    """Jacobian determinant statistics for a deformation field.

    Checks physical validity of a deformation (no folding).

    Parameters
    ----------
    flow : np.ndarray of shape (3, D, H, W)
        flow[0] = displacement along depth  (z)
        flow[1] = displacement along height (y)
        flow[2] = displacement along width  (x)

    Returns
    -------
    dict with keys: jac_mean, jac_std, jac_min, jac_max, jac_neg_pct

    Interpretation:
      > 1.0 = local expansion
      = 1.0 = no volume change
      0–1.0 = local compression
      < 0.0 = FOLDING (physically impossible, always bad)

    Classical reg: neg% should always be 0 %.
    VoxelMorph:    target < 1 %; use diffeomorphic variant if exceeded.
    """
    if flow.ndim != 4 or flow.shape[0] != 3:
        raise ValueError(f"flow must be (3, D, H, W), got {flow.shape}")

    dz = np.gradient(flow[0], axis=0)
    dy = np.gradient(flow[1], axis=1)
    dx = np.gradient(flow[2], axis=2)

    # Diagonal approximation of Jacobian determinant
    jac = (1 + dz) * (1 + dy) * (1 + dx)

    neg_pct = float((jac < 0).mean() * 100)
    return {
        "jac_mean":    round(float(jac.mean()), 4),
        "jac_std":     round(float(jac.std()),  4),
        "jac_min":     round(float(jac.min()),  4),
        "jac_max":     round(float(jac.max()),  4),
        "jac_neg_pct": round(neg_pct,            4),
    }


# ---------------------------------------------------------------------------
# Compute all metrics for one registered pair
# ---------------------------------------------------------------------------

def compute_all_metrics(
    mr_path:        str,
    ct_warped_path: str,
    mr_mask_path:   str,
    ct_mask_path:   str  = None,
    method:         str  = "affine",
) -> dict:
    """Compute all registration metrics for one subject.

    Handles missing files gracefully: missing CT mask sets dice/HD95 to None.

    Parameters
    ----------
    mr_path         : preprocessed MRI  (*_mr_norm.nii.gz)
    ct_warped_path  : registered CT     (*_ct_{method}.nii.gz)
    mr_mask_path    : brain mask in MRI space (*_mr_brain_mask.nii.gz)
    ct_mask_path    : CT brain mask, optional (*_ct_mask.nii.gz)
    method          : label for the registration method (rigid/affine/bspline)

    Returns
    -------
    dict with keys: method, dice, hd95, ncc, ct_mean_in_brain, ct_std_in_brain
    """
    import nibabel as nib
    import os

    result = {"method": method}

    try:
        mr_arr  = nib.load(mr_path).get_fdata().astype("float32")
        ct_arr  = nib.load(ct_warped_path).get_fdata().astype("float32")
        mr_mask = nib.load(mr_mask_path).get_fdata().astype(bool)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    # --- Dice + HD95 (need CT mask warped into MR space) ---
    if ct_mask_path and os.path.exists(ct_mask_path):
        try:
            ct_mask = nib.load(ct_mask_path).get_fdata().astype(bool)
            result["dice"] = round(dice_coefficient(mr_mask, ct_mask), 4)
            result["hd95"] = round(hausdorff95(mr_mask, ct_mask, voxel_size=1.0), 3)
        except Exception as exc:
            result["dice"] = None
            result["hd95"] = None
            result["dice_error"] = str(exc)
    else:
        result["dice"] = None
        result["hd95"] = None

    # --- NCC inside brain mask ---
    try:
        result["ncc"] = round(
            normalised_cross_correlation(mr_arr, ct_arr, mr_mask), 4
        )
    except Exception:
        result["ncc"] = None

    # --- Intensity stats of warped CT inside MRI brain mask ---
    try:
        ct_roi = ct_arr[mr_mask]
        result["ct_mean_in_brain"] = round(float(ct_roi.mean()), 4)
        result["ct_std_in_brain"]  = round(float(ct_roi.std()),  4)
    except Exception:
        pass

    return result
