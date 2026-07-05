"""
difference_maps.py
------------------
Computes difference maps between fixed MRI and registered (warped) CT.

After good registration, subtracting aligned images should show:
  - Near-zero values in correctly aligned healthy tissue
  - High values at registration errors (misalignment)
  - High values at true anatomical anomalies (future: lesion detection)

This is the FOUNDATION for Week 3's anomaly detection pipeline.
The same functions (with a trained VoxelMorph displacement field
instead of a classical registration result) become the anomaly detector.

Important: MRI and CT have different intensity scales
  MRI : z-score normalised (mean ~ 0)
  CT  : clipped and rescaled to [0, 1]
Raw subtraction is therefore meaningless.  Both volumes must be brought
onto a comparable scale BEFORE subtraction using percentile rescaling
inside the brain mask.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_logger

log = get_logger("difference_maps")


# ---------------------------------------------------------------------------
# Intensity matching before subtraction
# ---------------------------------------------------------------------------

def match_intensity_ranges(
    mr_arr: np.ndarray,
    ct_arr: np.ndarray,
    mask:   np.ndarray = None,
) -> tuple:
    """Rescale both volumes to [0, 1] using percentile normalisation.

    Percentiles are computed WITHIN the brain mask (if provided) to avoid
    bias from background voxels.  This puts MRI and CT on a comparable
    numerical scale before differencing.

    Note: this does NOT make MRI and CT semantically identical (they
    measure different tissue physics), but it ensures that large
    discrepancies (misalignment, gross anomalies) are numerically visible.

    Parameters
    ----------
    mr_arr, ct_arr : 3-D float arrays of identical shape
    mask           : optional binary brain mask (same shape)

    Returns
    -------
    (mr_rescaled, ct_rescaled) — both clipped to [0, 1]
    """
    def _rescale(arr: np.ndarray, m: np.ndarray) -> np.ndarray:
        roi = arr[m > 0] if m is not None else arr.ravel()
        if len(roi) == 0:
            return arr
        lo, hi = np.percentile(roi, 1), np.percentile(roi, 99)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    return _rescale(mr_arr, mask), _rescale(ct_arr, mask)


# ---------------------------------------------------------------------------
# Difference map computation
# ---------------------------------------------------------------------------

def compute_difference_map(
    mr_arr: np.ndarray,
    ct_arr: np.ndarray,
    mask:   np.ndarray = None,
    method: str        = "absolute",
) -> np.ndarray:
    """Compute a difference map between fixed MRI and warped CT.

    Intensities are matched to [0, 1] before subtraction.

    Parameters
    ----------
    mr_arr : fixed MRI array (z-score or raw — will be rescaled)
    ct_arr : warped CT array, same shape as mr_arr
    mask   : brain mask — difference is zeroed outside this region
    method : "absolute" for |a − b|, "signed" for (a − b)

    Returns
    -------
    diff : float32 array of same shape, zero outside brain mask
    """
    if mr_arr.shape != ct_arr.shape:
        raise ValueError(
            f"Shape mismatch: MR {mr_arr.shape} vs CT {ct_arr.shape}"
        )

    mr_s, ct_s = match_intensity_ranges(mr_arr, ct_arr, mask)

    if method == "absolute":
        diff = np.abs(mr_s - ct_s)
    elif method == "signed":
        diff = mr_s - ct_s
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'absolute' or 'signed'.")

    if mask is not None:
        diff = diff * (mask > 0).astype("float32")

    return diff.astype("float32")


def normalize_diff_by_local_std(
    diff: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Normalise a difference map by its own std inside the brain mask.

    Converts raw differences into a z-score-like map:
      'how many standard deviations above typical registration noise
       is each voxel?'

    This is what makes thresholding meaningful in Week 3:
      'this voxel is 3 std above typical registration noise' is a far
      stronger signal than a raw intensity difference.

    Parameters
    ----------
    diff : difference map (from compute_difference_map)
    mask : brain mask — std is computed inside this region only

    Returns
    -------
    diff_norm : same shape, divided by within-mask std
    """
    roi = diff[mask > 0] if mask is not None else diff[diff != 0]
    if len(roi) == 0:
        return diff
    std = roi.std() + 1e-8
    return diff / std


# ---------------------------------------------------------------------------
# Statistics on the difference map
# ---------------------------------------------------------------------------

def difference_map_stats(
    diff: np.ndarray,
    mask: np.ndarray = None,
) -> dict:
    """Summary statistics of a difference map.

    Used to compare registration quality across methods and to set
    anomaly-detection thresholds in Week 3.

    Parameters
    ----------
    diff : difference map array
    mask : restrict statistics to within-mask voxels

    Returns
    -------
    dict with keys: diff_mean, diff_std, diff_median, diff_p95, diff_max
    """
    roi = diff[mask > 0] if mask is not None else diff[diff != 0]

    if len(roi) == 0:
        return {
            "diff_mean":   0.0,
            "diff_std":    0.0,
            "diff_median": 0.0,
            "diff_p95":    0.0,
            "diff_max":    0.0,
        }

    return {
        "diff_mean":   round(float(roi.mean()),            5),
        "diff_std":    round(float(roi.std()),             5),
        "diff_median": round(float(np.median(roi)),        5),
        "diff_p95":    round(float(np.percentile(roi, 95)), 5),
        "diff_max":    round(float(roi.max()),             5),
    }


# ---------------------------------------------------------------------------
# Full pipeline: load → compute → save as NIfTI
# ---------------------------------------------------------------------------

def compute_and_save_difference_map(
    mr_path:   str,
    ct_path:   str,
    mask_path: str,
    out_path:  str,
    method:    str = "absolute",
) -> dict:
    """Load MRI + warped CT + mask, compute normalised diff, save as NIfTI.

    The saved volume is the std-normalised absolute difference map,
    directly usable as an anomaly score in Week 3.

    Parameters
    ----------
    mr_path   : preprocessed MRI  (*_mr_norm.nii.gz)
    ct_path   : registered CT     (*_ct_{method}.nii.gz)
    mask_path : brain mask        (*_mr_brain_mask.nii.gz), optional
    out_path  : output NIfTI path
    method    : "absolute" or "signed"

    Returns
    -------
    dict of difference map statistics (same as difference_map_stats)
    """
    import nibabel as nib

    mr_img  = nib.load(mr_path)
    ct_img  = nib.load(ct_path)
    mr_arr  = mr_img.get_fdata().astype("float32")
    ct_arr  = ct_img.get_fdata().astype("float32")

    mask_arr = None
    if mask_path and Path(mask_path).exists():
        mask_arr = nib.load(mask_path).get_fdata().astype("float32")

    diff      = compute_difference_map(mr_arr, ct_arr, mask_arr, method=method)
    diff_norm = normalize_diff_by_local_std(diff, mask_arr)

    # Save normalised diff as NIfTI (preserves MRI spatial metadata)
    diff_img = nib.Nifti1Image(diff_norm, mr_img.affine, mr_img.header)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(diff_img, out_path)

    stats = difference_map_stats(diff_norm, mask_arr)
    log.info(f"Diff map saved : {out_path}")
    log.info(f"  Stats        : {stats}")
    return stats
