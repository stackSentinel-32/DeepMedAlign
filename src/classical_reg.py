"""
classical_reg.py
----------------
Classical image registration pipeline using SimpleITK.

Three stages in exact order:
  Stage 1 -- Rigid    (6 DOF:  translation + rotation)
  Stage 2 -- Affine   (12 DOF: rigid + scaling + shear)
  Stage 3 -- B-spline (deformable: local non-linear deformations)

All stages use Mattes Mutual Information metric.
This is the ONLY valid similarity metric for MRI-CT cross-modal registration.
NCC, SSD, MSE all fail on cross-modal pairs.

Design notes
------------
MRI  = FIXED  image (reference — everything warps TO this)
CT   = MOVING image (warped into MRI space)

Why Mattes MI and NOT NCC:
  - NCC assumes a linear intensity relationship between modalities.
  - MRI bone = dark,  CT bone = bright (opposite!)
  - No linear relationship exists => NCC produces wrong deformation.
  - Mattes MI measures statistical dependency, not linear correlation.

Registration order — never skip stages:
  Rigid first  -> Affine on top -> B-spline on top.
  Each stage has a smaller capture range; jumping ahead loses convergence.

Usage:
  from src.classical_reg import register_full_pipeline
  result = register_full_pipeline(mr_path, ct_path, out_dir, subj_id)
"""

import sys
import time
import SimpleITK as sitk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_logger, timer

log = get_logger("classical_reg")


# ---------------------------------------------------------------------------
# Internal helper: build a registration method with Mattes MI
# ---------------------------------------------------------------------------

def _build_registration_method(
    fixed:        sitk.Image,
    moving:       sitk.Image,
    init_tx:      sitk.Transform,
    n_iters:      int   = 200,
    lr:           float = 1.0,
    bins:         int   = 50,
    sampling_pct: float = 0.20,
    shrink:       list  = None,
    sigma:        list  = None,
) -> sitk.ImageRegistrationMethod:
    """Build a SimpleITK registration method with Mattes MI and 3-level pyramid.

    Parameters
    ----------
    fixed, moving   : images already loaded as sitkFloat32
    init_tx         : initial transform (will NOT be modified in-place)
    n_iters         : gradient-descent iterations per resolution level
    lr              : learning rate
    bins            : number of histogram bins for Mattes MI
    sampling_pct    : fraction of voxels sampled per iteration (speed/accuracy)
    shrink, sigma   : multi-resolution pyramid factors and smoothing sigmas

    Returns
    -------
    sitk.ImageRegistrationMethod (not yet executed)
    """
    if shrink is None:
        shrink = [4, 2, 1]
    if sigma is None:
        sigma = [2.0, 1.0, 0.0]

    reg = sitk.ImageRegistrationMethod()

    # Metric: Mattes Mutual Information
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct)

    # Interpolator: bilinear during optimisation (fast)
    reg.SetInterpolator(sitk.sitkLinear)

    # Multi-resolution pyramid
    # Level 0: 4x shrink, 2 mm smooth  -- coarse, large capture range
    # Level 1: 2x shrink, 1 mm smooth  -- medium
    # Level 2: 1x shrink, 0 mm smooth  -- full resolution fine-tuning
    reg.SetShrinkFactorsPerLevel(shrink)
    reg.SetSmoothingSigmasPerLevel(sigma)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Optimizer: gradient descent with window-based convergence check
    reg.SetOptimizerAsGradientDescent(
        learningRate=lr,
        numberOfIterations=n_iters,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Initial transform is NOT modified in-place (safer for re-use)
    reg.SetInitialTransform(init_tx, inPlace=False)

    return reg


# ---------------------------------------------------------------------------
# Stage 1: Rigid registration
# ---------------------------------------------------------------------------

@timer
def register_rigid(
    fixed_path:  str,
    moving_path: str,
    out_path:    str,
    tx_path:     str,
    n_iters:     int   = 300,
    lr:          float = 1.0,
) -> sitk.Transform:
    """Rigid registration: 6 DOF (translation + rotation only).

    Corrects for gross positional differences between MRI and CT.
    Does NOT correct for scale or local deformations.
    Always run this FIRST; the result initialises the affine optimiser.

    Parameters
    ----------
    fixed_path  : path to preprocessed MRI (*_mr_norm.nii.gz)
    moving_path : path to preprocessed CT  (*_ct_norm.nii.gz)
    out_path    : where to save the rigidly-warped CT
    tx_path     : where to save the rigid transform (.tfm)
    n_iters     : iterations per pyramid level
    lr          : gradient-descent learning rate

    Returns
    -------
    sitk.Transform : the fitted rigid (Euler3D) transform
    """
    log.info("Rigid registration starting...")
    t0 = time.time()

    fixed  = sitk.ReadImage(str(fixed_path),  sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    log.info(f"  Fixed  (MR): {fixed.GetSize()}  @ {fixed.GetSpacing()} mm")
    log.info(f"  Moving (CT): {moving.GetSize()} @ {moving.GetSpacing()} mm")

    # Geometry-based initialisation: centres of bounding boxes.
    # Better than MOMENTS for images with very different intensity profiles.
    init_tx = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = _build_registration_method(
        fixed, moving, init_tx,
        n_iters=n_iters, lr=lr,
    )
    tx = reg.Execute(fixed, moving)

    # Resample CT into MRI space using the fitted transform
    warped = sitk.Resample(
        moving, fixed, tx,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(warped, str(out_path))
    sitk.WriteTransform(tx, str(tx_path))

    elapsed = time.time() - t0
    log.info(f"Rigid done in {elapsed:.1f}s  |  Final MI: {reg.GetMetricValue():.6f}")
    log.info(f"  Warped CT : {out_path}")
    log.info(f"  Transform : {tx_path}")
    return tx


# ---------------------------------------------------------------------------
# Stage 2: Affine registration
# ---------------------------------------------------------------------------

@timer
def register_affine(
    fixed_path:  str,
    moving_path: str,
    rigid_tx:    sitk.Transform,
    out_path:    str,
    tx_path:     str,
    n_iters:     int   = 300,
    lr:          float = 1.0,
) -> sitk.Transform:
    """Affine registration: 12 DOF (rigid + scaling + shear).

    Must be initialised from the rigid result.
    Running affine from scratch is unreliable due to its larger parameter space.

    Parameters
    ----------
    fixed_path  : path to preprocessed MRI
    moving_path : path to preprocessed CT (original, NOT the rigid-warped copy)
    rigid_tx    : fitted transform returned by register_rigid()
    out_path    : where to save the affinely-warped CT
    tx_path     : where to save the affine transform

    Returns
    -------
    sitk.Transform : the fitted affine transform
    """
    log.info("Affine registration starting (init from rigid)...")
    t0 = time.time()

    fixed  = sitk.ReadImage(str(fixed_path),  sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    # Copy rigid parameters into an AffineTransform
    affine = sitk.AffineTransform(3)
    if isinstance(rigid_tx, sitk.CompositeTransform):
        rigid_tx = sitk.Euler3DTransform(rigid_tx.GetNthTransform(0))
    affine.SetMatrix(rigid_tx.GetMatrix())
    affine.SetTranslation(rigid_tx.GetTranslation())
    affine.SetFixedParameters(rigid_tx.GetFixedParameters())

    reg = _build_registration_method(
        fixed, moving, affine,
        n_iters=n_iters, lr=lr,
    )
    tx = reg.Execute(fixed, moving)

    warped = sitk.Resample(
        moving, fixed, tx,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    sitk.WriteImage(warped, str(out_path))
    sitk.WriteTransform(tx, str(tx_path))

    elapsed = time.time() - t0
    log.info(f"Affine done in {elapsed:.1f}s  |  Final MI: {reg.GetMetricValue():.6f}")
    log.info(f"  Warped CT : {out_path}")
    log.info(f"  Transform : {tx_path}")
    return tx


# ---------------------------------------------------------------------------
# Stage 3: B-spline deformable registration
# ---------------------------------------------------------------------------

@timer
def register_bspline(
    fixed_path:  str,
    moving_path: str,
    affine_tx:   sitk.Transform,
    out_path:    str,
    tx_path:     str,
    mesh_size:   list = None,
) -> sitk.Transform:
    """B-spline deformable registration.

    Captures local non-linear deformations that affine cannot model:
      - Local brain shape differences between subjects
      - Partial-volume effects at tissue boundaries
      - Residual misalignments after affine correction

    This is the strongest classical baseline.
    VoxelMorph must beat this Dice score to be considered useful.

    The mesh_size controls the B-spline control-point grid density.
    Smaller grid = more local freedom = higher risk of overfitting.
    [8, 8, 8] is a sensible starting point for brain volumes.

    Uses the L-BFGS-B optimiser which is better suited for the large
    parameter space of B-spline transforms than gradient descent.

    Parameters
    ----------
    fixed_path  : path to preprocessed MRI
    moving_path : path to preprocessed CT (original)
    affine_tx   : fitted transform returned by register_affine()
    out_path    : where to save the deformably-warped CT
    tx_path     : where to save the composite transform
    mesh_size   : control-point grid [x, y, z] (default [8, 8, 8])

    Returns
    -------
    sitk.Transform : the fitted composite (affine + B-spline) transform
    """
    if mesh_size is None:
        mesh_size = [4, 4, 4]  # minimal grid: 192 vars; sufficient for a classical baseline

    log.info("B-spline registration starting (init from affine)...")
    log.info(f"  Control grid: {mesh_size}")
    t0 = time.time()

    fixed  = sitk.ReadImage(str(fixed_path),  sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    # Initialise B-spline grid from the fixed image geometry
    bspline_tx = sitk.BSplineTransformInitializer(
        image1=fixed,
        transformDomainMeshSize=mesh_size,
        order=3,
    )

    # KEY FIX: ITK metric-v4 can only compute the Jacobian of a single
    # optimizable transform.  Wrapping affine+bspline in a CompositeTransform
    # and passing it with inPlace=True triggers:
    #   "ComputeJacobianWithRespectToPosition is unimplemented for
    #    CompositeTransform"
    #
    # Correct pattern:
    #   SetMovingInitialTransform  -> affine (fixed, not optimized)
    #   SetInitialTransform        -> bspline only (the optimizable component)
    # SimpleITK internally concatenates both when resampling the moving image.

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20)
    reg.SetInterpolator(sitk.sitkLinear)
    # Single full-resolution pass: affine already handled coarse alignment.
    # Any pyramid level below full-res resamples the full volume again = wasted time.
    reg.SetShrinkFactorsPerLevel([1])
    reg.SetSmoothingSigmasPerLevel([0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetMetricSamplingPercentage(0.10)  # 10% of 4.9M voxels = 490K samples; fast + stable
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-4,
        numberOfIterations=30,              # enough for local refinement after affine
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=100,  # hard cap
        costFunctionConvergenceFactor=1e+7,
    )
    # Affine is a fixed pre-warp; B-spline is the only optimized transform
    reg.SetMovingInitialTransform(affine_tx)
    reg.SetInitialTransform(bspline_tx, inPlace=True)

    tx = reg.Execute(fixed, moving)

    # Compose affine + fitted bspline into a single output transform so the
    # saved .tfm and the resampled image both reflect the full correction.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(affine_tx)
    composite.AddTransform(tx)

    warped = sitk.Resample(
        moving, fixed, composite,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    sitk.WriteImage(warped, str(out_path))
    # Save only the B-spline part (a plain BSplineTransform) — serializable in .tfm.
    # affine_tx returned by Execute() is itself a CompositeTransform, so building
    # CompositeTransform(affine_tx, bspline) creates a nested composite that ITK
    # cannot write to any format.  The warped image already includes the full
    # affine+bspline correction; the .tfm is only kept for reproducibility.
    sitk.WriteTransform(tx, str(tx_path))

    elapsed = time.time() - t0
    log.info(f"B-spline done in {elapsed:.1f}s  |  Final MI: {reg.GetMetricValue():.6f}")
    log.info(f"  Warped CT : {out_path}")
    log.info(f"  Transform : {tx_path}")
    return composite


# ---------------------------------------------------------------------------
# Full pipeline: run all 3 stages for one subject
# ---------------------------------------------------------------------------

@timer
def register_full_pipeline(
    fixed_path:  str,
    moving_path: str,
    out_dir:     str,
    subj_id:     str,
    run_bspline: bool = True,
) -> dict:
    """Full classical registration pipeline for one subject.

    Executes: Rigid -> Affine -> B-spline (optional).

    Parameters
    ----------
    fixed_path  : preprocessed MRI path  (*_mr_norm.nii.gz)
    moving_path : preprocessed CT path   (*_ct_norm.nii.gz)
    out_dir     : output directory (typically data/processed/<subj_id>/)
    subj_id     : subject identifier string (e.g. '1BA001')
    run_bspline : whether to run the B-spline stage (slower, more accurate)

    Returns
    -------
    dict mapping output file keys to their absolute paths
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'=' * 55}")
    log.info(f"Classical registration: {subj_id}")
    log.info(f"{'=' * 55}")
    log.info(f"  Fixed  (MR): {fixed_path}")
    log.info(f"  Moving (CT): {moving_path}")

    # Stage 1: Rigid
    rigid_out = str(out / f"{subj_id}_ct_rigid.nii.gz")
    rigid_tx  = str(out / f"{subj_id}_rigid.tfm")
    r_tx = register_rigid(fixed_path, moving_path, rigid_out, rigid_tx)

    # Stage 2: Affine (initialised from rigid)
    affine_out = str(out / f"{subj_id}_ct_affine.nii.gz")
    affine_tx  = str(out / f"{subj_id}_affine.tfm")
    a_tx = register_affine(fixed_path, moving_path, r_tx, affine_out, affine_tx)

    result = {
        "ct_rigid":  rigid_out,
        "ct_affine": affine_out,
        "rigid_tx":  rigid_tx,
        "affine_tx": affine_tx,
    }

    # Stage 3: B-spline (optional, initialised from affine)
    if run_bspline:
        bspline_out = str(out / f"{subj_id}_ct_bspline.nii.gz")
        bspline_tx  = str(out / f"{subj_id}_bspline.tfm")
        register_bspline(fixed_path, moving_path, a_tx, bspline_out, bspline_tx)
        result["ct_bspline"] = bspline_out
        result["bspline_tx"] = bspline_tx

    log.info(f"\nPipeline complete for {subj_id}")
    log.info(f"  Outputs: {list(result.keys())}")
    return result
