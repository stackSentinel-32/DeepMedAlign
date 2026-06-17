"""
preprocess_mri.py
-----------------
Full MRI preprocessing pipeline for SynthRAD2023:
  1. Reorient to RAS+
  2. Resample to isotropic 1mm
  3. N4 bias field correction
  4. Skull stripping via HD-BET
  5. Z-score normalization inside brain mask
  6. Crop or pad to fixed shape

Each function is independently testable.
Never chain them blindly -- verify each step output first.
"""

import os
import subprocess
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import VOXEL_SPACING, FIXED_SHAPE
from src.utils import get_logger, ensure_dir, timer

log = get_logger("preprocess_mri")


# ── Step 1: Reorientation ──────────────────────────────────────────────────────

def reorient_to_ras(img: sitk.Image) -> sitk.Image:
    """
    Reorient image to RAS+ coordinate system.
    RAS = Right-Anterior-Superior axes.
    This standardises direction across all subjects.
    Must be the FIRST step -- always.
    """
    return sitk.DICOMOrient(img, "RAS")


# ── Step 2: Resampling ────────────────────────────────────────────────────────

def resample_isotropic(img: sitk.Image,
                       spacing: tuple = VOXEL_SPACING,
                       interp: int = sitk.sitkBSpline) -> sitk.Image:
    """
    Resample image to isotropic voxel spacing.
    Default: 1mm x 1mm x 1mm.
    Uses BSpline interpolation for MRI (smoother).
    Use sitkLinear for CT (avoids ringing on bone edges).
    """
    orig_sp  = img.GetSpacing()
    orig_sz  = img.GetSize()
    new_size = [
        int(round(orig_sz[i] * (orig_sp[i] / spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # explicit identity
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interp)

    result = resampler.Execute(img)
    log.info(f"Resampled: {orig_sz} @ {orig_sp}mm "
             f"-> {result.GetSize()} @ {spacing}mm")
    return result


# ── Step 3: N4 Bias Field Correction ─────────────────────────────────────────

def n4_bias_correction(img: sitk.Image,
                       n_iters: list = None) -> sitk.Image:
    """
    N4 bias field correction removes MRI intensity inhomogeneity
    caused by RF coil non-uniformity.
    Apply to MRI ONLY -- never to CT.
    Input must be float32 (cast automatically here).

    n_iters: iterations per resolution level.
             Default [50,50,30,20] is standard.
             Increase for heavily biased scans.
    """
    if n_iters is None:
        n_iters = [50, 50, 30, 20]

    img_f = sitk.Cast(img, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_iters)

    log.info("Running N4 bias correction...")
    result = corrector.Execute(img_f)
    log.info("N4 done.")
    return result


# ── Step 4: Skull Stripping ───────────────────────────────────────────────────

def skull_strip_mri(in_path: str,
                    out_path: str,
                    device: str = "cpu") -> str:
    """
    Run HD-BET skull stripping on MRI volume.
    Requires HD-BET installed: pip install hd-bet

    Returns: path to the brain mask file (_mask.nii.gz)
    Produces two outputs:
      out_path              -- skull-stripped MRI
      out_path_mask.nii.gz  -- binary brain mask (0/1)
    """
    log.info(f"Running HD-BET on: {in_path}")
    cmd = [
        "hd-bet",
        "-i", str(in_path),
        "-o", str(out_path),
        "-device", device,
        "-mode", "fast",
        "-tta", "0",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # HD-BET writes mask as: <stem>_mask.nii.gz
        mask_path = str(out_path).replace(".nii.gz", "_mask.nii.gz")
        log.info(f"HD-BET done. Mask: {mask_path}")
        return mask_path
    except subprocess.CalledProcessError as e:
        log.error(f"HD-BET failed: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        log.error("HD-BET not found. Install with: pip install hd-bet")
        raise


def skull_strip_fallback(img: sitk.Image,
                         out_mask_path: str = None) -> tuple:
    """
    Fallback skull strip using Otsu threshold + morphology.
    Use this if HD-BET is not available or fails.
    Less accurate than HD-BET but works without GPU.

    Returns: (skull_stripped_image, mask_image)
    """
    log.info("Using fallback skull strip (Otsu + morphology)...")

    # Cast to float
    img_f = sitk.Cast(img, sitk.sitkFloat32)

    # Otsu threshold
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = otsu.Execute(img_f)

    # Morphological closing to fill holes
    mask = sitk.BinaryMorphologicalClosing(mask, [5, 5, 5])

    # Keep largest connected component
    cc   = sitk.ConnectedComponent(mask)
    cc   = sitk.RelabelComponent(cc, sortByObjectSize=True)
    mask = sitk.Cast(sitk.Equal(cc, 1), sitk.sitkUInt8)

    # Apply mask
    mask_f   = sitk.Cast(mask, sitk.sitkFloat32)
    stripped = sitk.Multiply(img_f, mask_f)

    if out_mask_path:
        sitk.WriteImage(mask, str(out_mask_path))
        log.info(f"Fallback mask saved: {out_mask_path}")

    return stripped, mask


# ── Step 5: Normalization ─────────────────────────────────────────────────────

def zscore_normalize(img: sitk.Image,
                     mask: sitk.Image = None) -> sitk.Image:
    """
    Z-score normalization: (x - mean) / std
    Computed INSIDE brain mask only.
    After this: mean ~ 0.0, std ~ 1.0 inside mask.
    """
    arr = sitk.GetArrayFromImage(img).astype("float32")

    if mask is not None:
        m_arr = sitk.GetArrayFromImage(mask).astype(bool)
        roi   = arr[m_arr]
    else:
        roi = arr[arr != 0]

    if len(roi) == 0:
        log.warning("No non-zero voxels found for normalization")
        roi = arr.ravel()

    mu    = float(roi.mean())
    sigma = float(roi.std()) + 1e-8
    arr   = (arr - mu) / sigma

    log.info(f"Z-score: mean={mu:.2f} std={sigma:.2f} "
             f"-> normalized mean~{arr[arr != 0].mean():.3f}")

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


# ── Step 6: Crop / Pad ────────────────────────────────────────────────────────

def crop_or_pad(img: sitk.Image,
                target: tuple = FIXED_SHAPE) -> sitk.Image:
    """
    Crop or zero-pad volume to exact target shape (D, H, W).
    Padding is added at the end of each axis.
    Cropping removes from the end of each axis.
    This ensures all volumes have identical spatial dimensions.
    """
    arr = sitk.GetArrayFromImage(img).astype("float32")
    out = np.zeros(target, dtype="float32")
    s   = tuple(min(a, b) for a, b in zip(arr.shape, target))
    out[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]

    result = sitk.GetImageFromArray(out)
    result.SetSpacing(img.GetSpacing())
    result.SetDirection(img.GetDirection())
    result.SetOrigin(img.GetOrigin())

    log.info(f"crop_or_pad: {arr.shape} -> {out.shape}")
    return result


# ── Full pipeline ─────────────────────────────────────────────────────────────

@timer
def preprocess_mri_full(raw_path: str,
                        out_dir: str,
                        subj_id: str,
                        use_hdbet: bool = True,
                        device: str = "cpu") -> dict:
    """
    Full MRI preprocessing pipeline. Steps in order:
      1. Load
      2. Reorient to RAS
      3. Resample to 1mm isotropic
      4. N4 bias correction
      5. Skull strip (HD-BET or fallback)
      6. Z-score normalize inside brain mask
      7. Crop/pad to fixed shape
      8. Save all outputs

    Returns dict with paths to all saved files.
    """
    ensure_dir(out_dir)
    out = Path(out_dir)

    log.info(f"=== MRI pipeline: {subj_id} ===")

    # 1. Load
    img = sitk.ReadImage(str(raw_path))
    log.info(f"Loaded: {img.GetSize()} @ {img.GetSpacing()}mm")

    # 2. Reorient
    img = reorient_to_ras(img)

    # 3. Resample
    img = resample_isotropic(img, spacing=VOXEL_SPACING,
                              interp=sitk.sitkBSpline)

    # 4. N4 bias correction
    img = n4_bias_correction(img)

    # Save pre-stripped for HD-BET input (HD-BET needs a file path)
    prestrip_path = str(out / f"{subj_id}_mr_prestrip.nii.gz")
    sitk.WriteImage(img, prestrip_path)

    # 5. Skull strip
    mask_path  = str(out / f"{subj_id}_mr_brain_mask.nii.gz")
    brain_path = str(out / f"{subj_id}_mr_brain.nii.gz")

    if use_hdbet:
        try:
            returned_mask = skull_strip_mri(prestrip_path, brain_path,
                                            device=device)
            brain = sitk.ReadImage(brain_path)
            mask  = sitk.ReadImage(returned_mask)
            # Normalise mask path to our naming convention
            if returned_mask != mask_path:
                sitk.WriteImage(mask, mask_path)
        except Exception:
            log.warning("HD-BET failed, using fallback skull strip")
            brain, mask = skull_strip_fallback(img, mask_path)
            sitk.WriteImage(brain, brain_path)
    else:
        log.info("Skipping HD-BET, using fallback skull strip")
        brain, mask = skull_strip_fallback(img, mask_path)
        sitk.WriteImage(brain, brain_path)
        sitk.WriteImage(mask, mask_path)

    # Clean up temp pre-strip file
    if Path(prestrip_path).exists():
        os.remove(prestrip_path)

    # 6. Z-score normalize
    brain_norm = zscore_normalize(brain, mask)

    # 7. Crop/pad
    brain_norm = crop_or_pad(brain_norm, target=FIXED_SHAPE)
    mask       = crop_or_pad(mask,       target=FIXED_SHAPE)

    # 8. Save
    norm_path = str(out / f"{subj_id}_mr_norm.nii.gz")
    sitk.WriteImage(brain_norm, norm_path)
    sitk.WriteImage(mask, mask_path)

    log.info(f"=== MRI pipeline complete: {subj_id} ===")
    return {
        "mr_norm":  norm_path,
        "mr_mask":  mask_path,
        "mr_brain": brain_path,
    }
