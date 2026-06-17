"""
preprocess_ct.py
----------------
Full CT preprocessing pipeline for SynthRAD2023:
  1. Reorient to RAS+
  2. Resample to isotropic 1mm  (Linear interpolation -- no BSpline on CT)
  3. HU clipping  (-1000 → +1000  global,  -15 → +80 for brain window)
  4. Apply brain mask from paired MRI  (keeps only brain region)
  5. Min-max normalization to [0, 1]  inside mask
  6. Crop or pad to fixed shape

No N4 or skull stripping needed -- CT is quantitative by nature.
Each function is independently testable.
"""

import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import VOXEL_SPACING, FIXED_SHAPE, CT_HU_CLIP, CT_HU_BRAIN
from src.preprocess_mri import reorient_to_ras, resample_isotropic, crop_or_pad
from src.utils import get_logger, ensure_dir, timer

log = get_logger("preprocess_ct")


# ── Step 1+2: Reorient & Resample (reused from preprocess_mri) ────────────────
# reorient_to_ras()   -- imported above
# resample_isotropic()-- imported above, use sitkLinear for CT


# ── Step 3: HU Clipping ───────────────────────────────────────────────────────

def clip_hu(img: sitk.Image,
            hu_min: float = CT_HU_CLIP[0],
            hu_max: float = CT_HU_CLIP[1]) -> sitk.Image:
    """
    Clip CT Hounsfield Units to [hu_min, hu_max].
    Default: (-1000, +1000) — removes air artefacts and extreme bone.
    For brain-only CT use CT_HU_BRAIN = (-15, +80).

    Always apply BEFORE normalization.
    Do NOT apply N4 to CT -- it assumes smooth bias, not discrete HU.
    """
    arr = sitk.GetArrayFromImage(img).astype("float32")
    arr = np.clip(arr, hu_min, hu_max)
    log.info(f"HU clip: [{hu_min}, {hu_max}]  "
             f"actual range after clip: [{arr.min():.0f}, {arr.max():.0f}]")

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


# ── Step 4: Apply MRI brain mask ──────────────────────────────────────────────

def apply_brain_mask(ct_img: sitk.Image,
                     mask: sitk.Image) -> sitk.Image:
    """
    Zero-out CT voxels outside the MRI-derived brain mask.
    The mask must be in the same space as ct_img (same origin/spacing/size).
    If sizes differ (e.g. after independent resampling), resample mask first.

    Assumes mask is binary uint8 (0=background, 1=brain).
    """
    # Ensure mask matches CT geometry
    if ct_img.GetSize() != mask.GetSize():
        log.info("Resampling mask to match CT size...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        mask = resampler.Execute(mask)

    mask_f   = sitk.Cast(mask, sitk.sitkFloat32)
    ct_f     = sitk.Cast(ct_img, sitk.sitkFloat32)
    masked   = sitk.Multiply(ct_f, mask_f)

    brain_vox = sitk.GetArrayFromImage(masked)
    n_brain   = int((sitk.GetArrayFromImage(mask) > 0).sum())
    log.info(f"Brain mask applied: {n_brain} brain voxels retained, "
             f"range=[{brain_vox[brain_vox != 0].min():.0f}, "
             f"{brain_vox[brain_vox != 0].max():.0f}] HU")
    return masked


# ── Step 5: Min-Max Normalization ─────────────────────────────────────────────

def minmax_normalize(img: sitk.Image,
                     mask: sitk.Image = None,
                     hu_min: float = CT_HU_BRAIN[0],
                     hu_max: float = CT_HU_BRAIN[1]) -> sitk.Image:
    """
    Min-max normalize CT to [0, 1] using the brain HU window.
    Formula: (x - hu_min) / (hu_max - hu_min), then clip to [0, 1].

    Using a fixed window (CT_HU_BRAIN = -15..80 HU) instead of per-scan
    statistics ensures consistent contrast across all subjects.

    If mask provided, statistics are computed inside mask only
    (but normalization is applied globally).
    """
    arr = sitk.GetArrayFromImage(img).astype("float32")

    denom = (hu_max - hu_min) + 1e-8
    arr   = (arr - hu_min) / denom
    arr   = np.clip(arr, 0.0, 1.0)

    if mask is not None:
        m_arr   = sitk.GetArrayFromImage(mask).astype(bool)
        roi     = arr[m_arr]
        log.info(f"Min-max norm (window [{hu_min},{hu_max}] HU): "
                 f"brain mean={roi.mean():.3f}  std={roi.std():.3f}")
    else:
        log.info(f"Min-max norm (window [{hu_min},{hu_max}] HU): "
                 f"global mean={arr.mean():.3f}  std={arr.std():.3f}")

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


# ── Full pipeline ─────────────────────────────────────────────────────────────

@timer
def preprocess_ct_full(raw_path: str,
                       out_dir: str,
                       subj_id: str,
                       mask_path: str = None) -> dict:
    """
    Full CT preprocessing pipeline. Steps:
      1. Load
      2. Reorient to RAS
      3. Resample to 1mm isotropic (Linear -- NOT BSpline)
      4. Clip HU to (-1000, +1000)
      5. Apply brain mask (from paired MRI preprocessing, if available)
      6. Clip to brain HU window (-15, +80)
      7. Min-max normalize to [0, 1]
      8. Crop/pad to fixed shape
      9. Save outputs

    Args:
        raw_path:   Path to raw CT NIfTI (.nii.gz)
        out_dir:    Output directory
        subj_id:    Subject identifier string
        mask_path:  Path to MRI-derived brain mask (optional but recommended)

    Returns dict of output file paths.
    """
    ensure_dir(out_dir)
    out = Path(out_dir)

    log.info(f"=== CT pipeline: {subj_id} ===")

    # 1. Load
    img = sitk.ReadImage(str(raw_path))
    log.info(f"Loaded: {img.GetSize()} @ {img.GetSpacing()}mm  "
             f"pixel type={img.GetPixelIDTypeAsString()}")

    # 2. Reorient to RAS
    img = reorient_to_ras(img)

    # 3. Resample — Linear interpolation (BSpline causes ringing on CT)
    img = resample_isotropic(img, spacing=VOXEL_SPACING,
                             interp=sitk.sitkLinear)

    # 4. Global HU clip
    img = clip_hu(img, hu_min=CT_HU_CLIP[0], hu_max=CT_HU_CLIP[1])

    # 5. Apply brain mask (if available)
    mask = None
    if mask_path and Path(mask_path).exists():
        mask = sitk.ReadImage(str(mask_path))
        img  = apply_brain_mask(img, mask)
    else:
        log.warning("No brain mask provided — CT not masked. "
                    "Run MRI pipeline first to generate mask.")

    # 6. Brain window HU clip
    img = clip_hu(img, hu_min=CT_HU_BRAIN[0], hu_max=CT_HU_BRAIN[1])

    # 7. Min-max normalize
    img = minmax_normalize(img, mask=mask,
                           hu_min=CT_HU_BRAIN[0], hu_max=CT_HU_BRAIN[1])

    # 8. Crop/pad
    img = crop_or_pad(img, target=FIXED_SHAPE)
    if mask is not None:
        mask = crop_or_pad(mask, target=FIXED_SHAPE)

    # 9. Save
    ct_norm_path = str(out / f"{subj_id}_ct_norm.nii.gz")
    sitk.WriteImage(img, ct_norm_path)
    log.info(f"Saved CT norm: {ct_norm_path}")

    result = {"ct_norm": ct_norm_path}

    if mask is not None:
        ct_mask_path = str(out / f"{subj_id}_ct_mask.nii.gz")
        sitk.WriteImage(mask, ct_mask_path)
        result["ct_mask"] = ct_mask_path

    log.info(f"=== CT pipeline complete: {subj_id} ===")
    return result
