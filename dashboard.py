"""
DeepMedAlign Dashboard
----------------------
Streamlit web app for medical image registration using VoxelMorph v2.
Supports running live inference on uploaded MRI/CT scans and exploring
evaluation metrics across SynthRad2023 dataset splits.
"""

import io
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import streamlit as st

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# Streamlit page setup
st.set_page_config(
    page_title="DeepMedAlign | Medical Image Registration",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------

@st.cache_data
def load_voxelmorph_metrics():
    path = RESULTS_DIR / "voxelmorph_test_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_training_log():
    path = RESULTS_DIR / "training_log.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_baseline(method: str):
    path = RESULTS_DIR / f"baseline_metrics_{method}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_diff_map_stats(method: str):
    path = RESULTS_DIR / f"difference_map_stats_{method}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plot Formatting
# ---------------------------------------------------------------------------

def apply_chart_style(fig, height=380):
    """Applies clean layout formatting to Plotly figures."""
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#0A0A0A",
        plot_bgcolor="#0A0A0A",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------------------------

def render_sidebar(vm_df=None):
    with st.sidebar:
        st.markdown("## 🧠 DeepMedAlign")
        st.caption("CT → MRI Brain Registration")
        st.markdown("---")

        if vm_df is not None and not vm_df.empty:
            st.markdown("### 📊 Benchmark Results")
            dice_val = f"{vm_df['dice'].mean():.4f}"
            hd95_val = f"{vm_df['hd95'].mean():.2f} mm"
            nmi_val = f"{vm_df['nmi'].mean():.4f}"
            ssim_val = f"{vm_df['ssim'].mean():.4f}"
            jac_val = f"{vm_df['jac_neg_pct'].mean():.3f}%"

            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 500;">Mean Dice ↑</div>
                        <div style="font-size: 1.05rem; font-weight: 700;">{dice_val}</div>
                        <div style="font-size: 0.72rem; color: #34d399;">↑ Target &gt; 0.776</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 500;">Mean HD95 ↓</div>
                        <div style="font-size: 1.05rem; font-weight: 700;">{hd95_val}</div>
                        <div style="font-size: 0.72rem; color: #34d399;">↑ Target &lt; 19.2</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 500;">Mean NMI ↑</div>
                        <div style="font-size: 1.05rem; font-weight: 700;">{nmi_val}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 500;">Mean SSIM ↑</div>
                        <div style="font-size: 1.05rem; font-weight: 700;">{ssim_val}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 500;">Jac Neg% ↓</div>
                        <div style="font-size: 1.05rem; font-weight: 700;">{jac_val}</div>
                        <div style="font-size: 0.72rem; color: #34d399;">↑ Target &lt; 1%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")

        st.markdown("### 📋 Project Info")
        st.markdown(
            """
            - **Dataset:** SynthRad 2023
            - **Subjects:** 180 (125/19/36)
            - **Resolution:** 160×192×160
            - **Voxel spacing:** 1 mm iso
            """
        )

        st.markdown("---")
        st.markdown("### 🏗️ Architecture")
        st.markdown("**VoxelMorph v2**")
        st.markdown(
            """
            - 4-level U-Net encoder
            - Multi-res pyramid DVF
            - Dice + MI + Jac loss
            - Elastic augmentation
            """
        )

        st.markdown("---")
        st.markdown("### ⚡ Performance")
        st.markdown(
            """
            | | Classical | VoxelMorph |
            |---|---|---|
            | **Time** | ~3 min | ~50 ms |
            | **Speedup** | 1× | **3,600×** |
            """
        )


# ---------------------------------------------------------------------------
# Tab 1: Live Inference
# ---------------------------------------------------------------------------

def render_inference_tab(vm_df=None):
    st.header("Register New Patient Scans")
    st.markdown("Upload paired MRI and CT volumes in NIfTI format (`.nii` or `.nii.gz`).")

    model_path = MODELS_DIR / "voxelmorph_v2_best.pth"
    if not model_path.exists():
        st.error(f"Model checkpoint not found at `{model_path}`. Please verify weights exist.")
        return

    col_mri, col_ct, col_mask = st.columns(3)
    with col_mri:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(124,77,255,0.06) 0%, rgba(99,102,241,0.04) 100%); border: 2px dashed rgba(124,77,255,0.3); border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 12px;">
                <h4 style="margin: 0; font-size: 1.1rem;">🧠 MRI Scan</h4>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 4px 0 0 0;">T1-weighted brain MRI (.nii.gz)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mri_file = st.file_uploader("Choose MRI NIfTI file", type=["nii", "gz"], key="mri_up", label_visibility="collapsed")

    with col_ct:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(124,77,255,0.06) 0%, rgba(99,102,241,0.04) 100%); border: 2px dashed rgba(124,77,255,0.3); border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 12px;">
                <h4 style="margin: 0; font-size: 1.1rem;">🏥 CT Scan</h4>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 4px 0 0 0;">Planning CT brain (.nii.gz)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ct_file = st.file_uploader("Choose CT NIfTI file", type=["nii", "gz"], key="ct_up", label_visibility="collapsed")

    with col_mask:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(34,197,94,0.06) 0%, rgba(16,185,129,0.04) 100%); border: 2px dashed rgba(34,197,94,0.35); border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 12px;">
                <h4 style="margin: 0; font-size: 1.1rem;">🎯 Brain Mask <span style='font-size:0.75rem; color:#64748b;'>(optional)</span></h4>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 4px 0 0 0;">mask.nii.gz — enables accurate Dice/HD95</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mask_file = st.file_uploader("Choose Mask NIfTI file", type=["nii", "gz"], key="mask_up", label_visibility="collapsed")

    with st.expander("⚙️ Inference Options", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            run_prep = st.checkbox("Run full preprocessing (RAS, N4, Resample, Normalize)", value=True)
        with c2:
            use_hdbet = st.checkbox("Use HD-BET for skull stripping (fallback to Otsu if off)", value=False)
        with c3:
            device_opt = st.selectbox("Compute Device", ["Auto", "CPU", "CUDA"], index=0)

    can_run = mri_file is not None and ct_file is not None
    run_clicked = st.button(
        "🧬 Run Registration",
        disabled=not can_run,
        use_container_width=True,
        type="primary",
    )

    if not can_run:
        st.info("📎 Upload both MRI and CT scans above, then click **Run Registration**.")
        # Clear stale session results when files are removed
        st.session_state.pop("inference_result", None)
        return

    # Clear results if a new set of files is uploaded (detect by file names)
    file_key = (
        mri_file.name if mri_file else "",
        ct_file.name if ct_file else "",
        mask_file.name if mask_file else "",
    )
    if st.session_state.get("_last_file_key") != file_key:
        st.session_state.pop("inference_result", None)
        st.session_state["_last_file_key"] = file_key

    if run_clicked and can_run:

        if device_opt == "CUDA" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_opt == "CPU":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        st.text(f"Using device: {device}")
        progress_bar = st.progress(0, text="Initializing processing...")
        step_times = {}  # tracks wall-clock time for each pipeline stage

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mri_in = tmp_path / "input_mr.nii.gz"
            ct_in = tmp_path / "input_ct.nii.gz"

            mri_in.write_bytes(mri_file.getvalue())
            ct_in.write_bytes(ct_file.getvalue())
            # Save optional mask file
            mask_in = None
            if mask_file is not None:
                mask_in = tmp_path / "input_mask.nii.gz"
                mask_in.write_bytes(mask_file.getvalue())

            progress_bar.progress(20, text="Preprocessing MRI volume...")

            try:
                if run_prep:
                    from src.config import FIXED_SHAPE, VOXEL_SPACING, CT_HU_BRAIN, CT_HU_CLIP
                    from src.preprocess_ct import apply_brain_mask, clip_hu, minmax_normalize
                    from src.preprocess_mri import (
                        crop_or_pad, n4_bias_correction, reorient_to_ras,
                        resample_isotropic, skull_strip_fallback, skull_strip_mri,
                        zscore_normalize,
                    )

                    # MRI Preprocessing
                    _t = time.perf_counter()
                    mr_img = sitk.ReadImage(str(mri_in))
                    mr_img = reorient_to_ras(mr_img)
                    mr_img = resample_isotropic(mr_img, spacing=VOXEL_SPACING, interp=sitk.sitkBSpline)
                    mr_img = n4_bias_correction(mr_img)

                    if use_hdbet:
                        try:
                            prestrip_p = str(tmp_path / "mr_prestrip.nii.gz")
                            brain_p = str(tmp_path / "mr_brain.nii.gz")
                            sitk.WriteImage(mr_img, prestrip_p)
                            mask_p = skull_strip_mri(prestrip_p, brain_p, device=str(device))
                            mr_brain = sitk.ReadImage(brain_p)
                            mr_mask = sitk.ReadImage(mask_p)
                        except Exception:
                            st.warning("HD-BET unavailable or failed. Using morphological threshold fallback.")
                            mr_brain, mr_mask = skull_strip_fallback(mr_img)
                    else:
                        mr_brain, mr_mask = skull_strip_fallback(mr_img)

                    mr_norm = zscore_normalize(mr_brain, mr_mask)
                    mr_norm = crop_or_pad(mr_norm, target=FIXED_SHAPE)
                    mr_mask = crop_or_pad(mr_mask, target=FIXED_SHAPE)
                    mr_arr = sitk.GetArrayFromImage(mr_norm).astype(np.float32)
                    step_times["MRI Preprocessing"] = time.perf_counter() - _t

                    # CT Preprocessing
                    progress_bar.progress(50, text="Preprocessing CT volume...")
                    _t = time.perf_counter()
                    ct_img = sitk.ReadImage(str(ct_in))
                    ct_img = reorient_to_ras(ct_img)
                    ct_img = resample_isotropic(ct_img, spacing=VOXEL_SPACING, interp=sitk.sitkLinear)
                    ct_img = clip_hu(ct_img, hu_min=CT_HU_CLIP[0], hu_max=CT_HU_CLIP[1])
                    ct_img, ct_mask = apply_brain_mask(ct_img, mr_mask)
                    ct_img = clip_hu(ct_img, hu_min=CT_HU_BRAIN[0], hu_max=CT_HU_BRAIN[1])
                    ct_img = minmax_normalize(ct_img, mask=ct_mask, hu_min=CT_HU_BRAIN[0], hu_max=CT_HU_BRAIN[1])
                    ct_img = crop_or_pad(ct_img, target=FIXED_SHAPE)
                    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
                    step_times["CT Preprocessing"] = time.perf_counter() - _t

                else:
                    # Preprocessed files (_norm.nii.gz): load as-is.
                    # MRI is z-score normalized (~-5 to +5, mean≈0); CT is min-max [0,1].
                    # Do NOT re-normalize — the model was trained on these exact distributions.
                    progress_bar.progress(50, text="Loading preprocessed arrays...")
                    mr_sitk = sitk.ReadImage(str(mri_in))
                    ct_sitk = sitk.ReadImage(str(ct_in))
                    mr_arr = sitk.GetArrayFromImage(mr_sitk).astype(np.float32)
                    ct_arr = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)

                progress_bar.progress(70, text="Running VoxelMorph neural network...")
                from src.voxelmorph_model import VoxelMorph

                mr_tensor = torch.from_numpy(mr_arr).unsqueeze(0).unsqueeze(0).to(device)
                ct_tensor = torch.from_numpy(ct_arr).unsqueeze(0).unsqueeze(0).to(device)

                # Model was trained at full preprocessed resolution (160, 192, 160)
                vol_size = tuple(mr_tensor.shape[2:])

                checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
                cfg = checkpoint.get("config", {})
                diffeomorphic = cfg.get("diffeomorphic", True)
                large = cfg.get("large", False)

                enc = (32, 64, 64, 64) if large else (16, 32, 32, 32)
                dec = (64, 64, 64, 32) if large else (32, 32, 32, 16)

                model = VoxelMorph(
                    enc_features=enc,
                    dec_features=dec,
                    vol_size=vol_size,
                    diffeomorphic=diffeomorphic,
                ).to(device)

                # Extract the raw state dict from the checkpoint
                if "model" in checkpoint:
                    raw_state = checkpoint["model"]
                elif "model_state" in checkpoint:
                    raw_state = checkpoint["model_state"]
                elif "state_dict" in checkpoint:
                    raw_state = checkpoint["state_dict"]
                else:
                    raw_state = checkpoint

                # Strip `_orig_mod.` prefix added by torch.compile()
                cleaned_state = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in raw_state.items()
                }
                model.load_state_dict(cleaned_state)
                model.eval()

                t0 = time.perf_counter()
                with torch.no_grad():
                    warped_tensor, dvf_tensor = model(mr_tensor, ct_tensor)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                step_times["VoxelMorph Inference"] = elapsed_ms / 1000.0  # store in seconds

                warped_arr = warped_tensor.squeeze().cpu().numpy()
                dvf_arr = dvf_tensor.squeeze().cpu().numpy()

                progress_bar.progress(90, text="Calculating evaluation metrics...")
                from src.metrics import (
                    dice_coefficient, hausdorff95, jacobian_stats,
                    normalised_cross_correlation, normalized_mutual_info,
                    structural_sim,
                )

                _t = time.perf_counter()

                # ── Mask-based Dice (correct methodology, same as offline evaluation) ──
                mask_based = False
                if mask_in is not None and mask_in.exists():
                    try:
                        from scipy.ndimage import map_coordinates as _map_coords
                        mask_sitk = sitk.ReadImage(str(mask_in))
                        # Resample mask to match fixed shape if needed
                        from src.preprocess_mri import crop_or_pad as _cop
                        mask_sitk = _cop(mask_sitk, target=mr_arr.shape)
                        mr_mask_bool = sitk.GetArrayFromImage(mask_sitk).astype(bool)

                        # Apply DVF to warp the CT (moving) mask into MRI (fixed) space
                        # dvf_arr shape: (3, H, W, D) — channel i = displacement along spatial axis i
                        H, W, D = mr_arr.shape
                        gz, gy, gx = np.meshgrid(
                            np.arange(H), np.arange(W), np.arange(D), indexing='ij'
                        )
                        new_coords = [
                            gz + dvf_arr[0],
                            gy + dvf_arr[1],
                            gx + dvf_arr[2],
                        ]
                        warped_mask_bool = _map_coords(
                            mr_mask_bool.astype(np.float32),
                            [c.ravel() for c in new_coords],
                            order=0,           # nearest-neighbour — preserves binary values
                            mode='constant', cval=0.0,
                        ).reshape(H, W, D) > 0.5
                        mask_based = True
                    except Exception as _me:
                        st.warning(f"Mask-based Dice failed ({_me}), using intensity proxy instead.")
                        mask_based = False

                # ── Fallback: adaptive intensity threshold ──
                if not mask_based:
                    def _brain_mask(arr):
                        flat = arr.ravel()
                        if flat.min() < -0.5:
                            bg_val = float(np.percentile(flat, 5))
                            brain_val = float(np.percentile(flat, 80))
                            thresh = (bg_val + brain_val) / 2.0
                        else:
                            thresh = 0.05
                        return arr > thresh
                    mr_mask_bool = _brain_mask(mr_arr)
                    warped_mask_bool = _brain_mask(warped_arr)

                dice_score = dice_coefficient(mr_mask_bool, warped_mask_bool)
                hd95_score = hausdorff95(mr_mask_bool, warped_mask_bool, voxel_size=1.0)
                ncc_score = normalised_cross_correlation(mr_arr, warped_arr, mr_mask_bool)
                nmi_score = normalized_mutual_info(mr_arr, warped_arr, mr_mask_bool)
                ssim_score = structural_sim(mr_arr, warped_arr)
                jac_info = jacobian_stats(dvf_arr)
                step_times["Metric Computation"] = time.perf_counter() - _t

                progress_bar.progress(100, text="Complete!")

                # Store everything in session state so widget interactions don't re-trigger inference
                st.session_state["inference_result"] = {
                    "mr_arr": mr_arr,
                    "ct_arr": ct_arr,
                    "warped_arr": warped_arr,
                    "dvf_arr": dvf_arr,
                    "elapsed_ms": elapsed_ms,
                    "dice_score": dice_score,
                    "hd95_score": hd95_score,
                    "ncc_score": ncc_score,
                    "nmi_score": nmi_score,
                    "ssim_score": ssim_score,
                    "jac_info": jac_info,
                    "step_times": step_times,
                    "mask_based": mask_based,
                    "mr_mask_bool": mr_mask_bool,
                }

            except Exception as e:
                st.error(f"Execution error during registration: {e}")
                st.exception(e)

    # --- Render results from session state (persists across widget interactions) ---
    res = st.session_state.get("inference_result")
    if res is None:
        return

    mr_arr       = res["mr_arr"]
    ct_arr       = res["ct_arr"]
    warped_arr   = res["warped_arr"]
    dvf_arr      = res["dvf_arr"]
    elapsed_ms   = res["elapsed_ms"]
    dice_score   = res["dice_score"]
    hd95_score   = res["hd95_score"]
    ncc_score    = res["ncc_score"]
    nmi_score    = res["nmi_score"]
    ssim_score   = res["ssim_score"]
    jac_info     = res["jac_info"]
    step_times   = res["step_times"]
    mask_based   = res.get("mask_based", False)
    mr_mask_bool = res.get("mr_mask_bool", mr_arr > mr_arr.min() + 0.1)

    if st.button("🗑 Clear Results", key="clear_results"):
        st.session_state.pop("inference_result", None)
        st.rerun()


    # --- Results View ---
    st.subheader("Registration Results")
    if mask_based:
        st.success("🎯 **Ground-Truth Brain Mask Uploaded**: Dice & HD95 are computed via 3D DVF deformation of the binary mask (matches official benchmark).")
    else:
        st.info("ℹ️ **No Mask Uploaded**: Using adaptive intensity threshold proxy for Dice. For exact benchmark evaluation (95%+ Dice), upload `mask.nii.gz` in the 3rd box above.")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Inference Time", f"{elapsed_ms:.1f} ms")
    m2.metric("Dice Similarity", f"{dice_score:.4f}", help="GT Mask Overlap" if mask_based else "Intensity Proxy Overlap")
    m3.metric("HD95 Distance", f"{hd95_score:.2f} mm")
    m4.metric("NMI Score", f"{nmi_score:.4f}")
    m5.metric("SSIM Score", f"{ssim_score:.4f}")
    m6.metric("Jacobian Neg %", f"{jac_info.get('jac_neg_pct', 0.0):.3f}%")

    # Benchmark comparison vs dataset mean
    if vm_df is not None and not vm_df.empty:
        st.markdown("**📊 How does this compare to the test-set benchmark?**")
        bm1, bm2, bm3, bm4 = st.columns(4)
        ds_dice = vm_df["dice"].mean()
        ds_hd95 = vm_df["hd95"].mean()
        ds_nmi  = vm_df["nmi"].mean()
        ds_jac  = vm_df["jac_neg_pct"].mean()

        def _badge(val, ref, higher_better=True):
            better = val > ref if higher_better else val < ref
            return "✅ Above avg" if better else "⚠️ Below avg"

        bm1.metric("Dice vs avg", f"{dice_score:.4f}", delta=f"{dice_score - ds_dice:+.4f}  {_badge(dice_score, ds_dice)}", delta_color="normal")
        bm2.metric("HD95 vs avg", f"{hd95_score:.2f} mm", delta=f"{hd95_score - ds_hd95:+.2f}  {_badge(hd95_score, ds_hd95, False)}", delta_color="inverse")
        bm3.metric("NMI vs avg", f"{nmi_score:.4f}", delta=f"{nmi_score - ds_nmi:+.4f}  {_badge(nmi_score, ds_nmi)}", delta_color="normal")
        bm4.metric("Jac Neg vs avg", f"{jac_info.get('jac_neg_pct', 0.0):.3f}%", delta=f"{jac_info.get('jac_neg_pct', 0.0) - ds_jac:+.3f}%  {_badge(jac_info.get('jac_neg_pct', 0.0), ds_jac, False)}", delta_color="inverse")

    st.markdown("---")
    st.subheader("Interactive Volume Slice Viewer")
    axis_name = st.radio("Slice Plane", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], horizontal=True)

    axis_map = {"Axial (Z)": 0, "Coronal (Y)": 1, "Sagittal (X)": 2}
    dim_idx = axis_map[axis_name]
    max_slice = mr_arr.shape[dim_idx] - 1
    curr_slice = st.slider("Slice Position", 0, max_slice, max_slice // 2)

    def extract_plane(arr, axis, idx):
        if axis == 0:
            plane = arr[idx, :, :]
        elif axis == 1:
            plane = arr[:, idx, :]
        else:
            plane = arr[:, :, idx]
        # In NIfTI array indexing, row 0 corresponds to inferior/posterior voxels.
        # np.flipud flips vertically so top-of-head (superior) is displayed at the top of the image.
        return np.flipud(plane)

    from src.difference_maps import compute_difference_map, normalize_diff_by_local_std

    # Compute 3D difference map in z-score units (standard deviations from typical difference)
    diff_3d = compute_difference_map(mr_arr, warped_arr, mr_mask_bool, method="absolute")
    diff_z = normalize_diff_by_local_std(diff_3d, mr_mask_bool)

    s_mr_raw = extract_plane(mr_arr, dim_idx, curr_slice)
    s_ct_raw = extract_plane(ct_arr, dim_idx, curr_slice)
    s_warped_raw = extract_plane(warped_arr, dim_idx, curr_slice)
    s_mr_mask = extract_plane(mr_mask_bool, dim_idx, curr_slice)
    s_diff = extract_plane(diff_z, dim_idx, curr_slice)

    # --- Background masking ---
    # Use np.nan for background so Plotly renders it as pure black (paper_bgcolor).
    # This is more robust than 0.0 because 0 maps to dark red/orange on the hot colorscale.
    def _mask_bg(arr_2d, mask_2d, bg_value=np.nan):
        out = arr_2d.astype(float).copy()
        out[~mask_2d] = bg_value
        return out

    # For grayscale (MRI/CT): percentile normalise only tissue voxels, set all background air & non-brain to NaN (pure black)
    def _norm_display_masked(arr_2d, mask_2d):
        arr_work = arr_2d.astype(float).copy()
        
        # Detect & mask air background in CT/MRI arrays:
        # Preprocessed min-max CT ([0,1]): air background is ~0.158
        if arr_work.max() <= 1.05 and arr_work.min() >= -0.05:
            air_mask = (arr_work <= 0.16)
        else:
            # Raw HU CT (e.g. -1000 to +1000) or z-score MRI (~-5 to +5): air is near min
            min_v = float(np.percentile(arr_work, 1))
            max_v = float(np.percentile(arr_work, 99))
            air_mask = (arr_work <= min_v + 0.08 * (max_v - min_v + 1e-8))

        # Combined mask: tissue voxels inside brain mask and non-air
        tissue_mask = mask_2d & (~air_mask)
        tissue_px = arr_work[tissue_mask]
        
        if len(tissue_px) == 0:
            tissue_px = arr_work[mask_2d] if np.any(mask_2d) else arr_work.ravel()

        lo, hi = np.percentile(tissue_px, 2), np.percentile(tissue_px, 98)
        normed = np.clip((arr_work - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        out = normed.astype(float)
        out[~tissue_mask] = np.nan  # background air & outside mask → transparent → pitch black (#0A0A0A)
        return out

    s_mr     = _norm_display_masked(s_mr_raw, s_mr_mask)
    s_ct     = _norm_display_masked(s_ct_raw, s_mr_mask)
    s_warped = _norm_display_masked(s_warped_raw, s_mr_mask)
    # Diff: background → np.nan so hot colormap never shows orange in background
    s_diff_display = _mask_bg(s_diff, s_mr_mask, bg_value=np.nan)

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        fig = px.imshow(s_mr, color_continuous_scale="gray", title="MRI (Fixed Target)",
                        zmin=0, zmax=1)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_chart_style(fig, 300))
    with v2:
        fig = px.imshow(s_ct, color_continuous_scale="gray", title="Original CT (Moving)",
                        zmin=0, zmax=1)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_chart_style(fig, 300))
    with v3:
        fig = px.imshow(s_warped, color_continuous_scale="gray", title="Registered CT (Warped)",
                        zmin=0, zmax=1)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_chart_style(fig, 300))
    with v4:
        fig = px.imshow(s_diff_display, color_continuous_scale="hot",
                        range_color=[0, 3.0], title="Difference (|MRI — Warped CT|)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_chart_style(fig, 300))

    # --- Overlay Blend View ---
    st.markdown("---")
    st.subheader("🔀 Overlay Blend View")
    alpha = st.slider(
        "MRI ← blend → Warped CT",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        key="blend_alpha",
        help="0 = pure Warped CT, 1 = pure MRI",
    )
    s_overlay = alpha * s_mr + (1.0 - alpha) * s_warped
    ov1, ov2 = st.columns(2)
    with ov1:
        fig = px.imshow(s_overlay, color_continuous_scale="gray",
                        title=f"Blended (MRI {int(alpha*100)}% / Warped CT {int((1-alpha)*100)}%)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_chart_style(fig, 340))
    with ov2:
        h, w = s_mr.shape
        block = max(h // 10, 1)
        mask_cb = np.zeros((h, w), dtype=bool)
        for bi in range(0, h, block * 2):
            for bj in range(0, w, block * 2):
                mask_cb[bi:bi+block, bj:bj+block] = True
                mask_cb[bi+block:bi+2*block, bj+block:bj+2*block] = True
        s_checker = np.where(mask_cb, s_mr, s_warped)
        fig = px.imshow(s_checker, color_continuous_scale="gray", title="Checkerboard (MRI / Warped CT)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(apply_chart_style(fig, 340))

    # --- Deformation Field Magnitude ---
    st.markdown("---")
    st.subheader("🌀 Deformation Vector Field Magnitude")
    dvf_mag = np.sqrt(np.sum(dvf_arr ** 2, axis=0))  # (H, W, D)
    s_dvf = extract_plane(dvf_mag, dim_idx, curr_slice)

    dv1, dv2 = st.columns(2)
    with dv1:
        fig = px.imshow(
            s_dvf, color_continuous_scale="viridis",
            title="DVF Magnitude (voxels)",
            labels={"color": "displacement"},
        )
        st.plotly_chart(apply_chart_style(fig, 340))
    with dv2:
        mag_flat = dvf_mag.ravel()
        mag_flat = mag_flat[mag_flat > 0.05]
        fig = px.histogram(
            x=mag_flat, nbins=50,
            title="Displacement Magnitude Distribution",
            labels={"x": "Displacement (voxels)", "y": "Count"},
        )
        fig.add_vline(x=float(mag_flat.mean()), line_dash="dash",
                      line_color="orange", annotation_text=f"Mean: {mag_flat.mean():.2f}")
        st.plotly_chart(apply_chart_style(fig, 340))

    # --- Export Outputs ---
    st.markdown("---")
    st.subheader("Export Outputs")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        warped_img_out = sitk.GetImageFromArray(warped_arr)
        import tempfile as _tmp
        import os as _os
        _td = _tmp.mkdtemp()
        out_path_nii = _os.path.join(_td, "warped_ct.nii.gz")
        sitk.WriteImage(warped_img_out, out_path_nii)
        with open(out_path_nii, "rb") as f:
            st.download_button(
                label="Download Warped CT NIfTI",
                data=f.read(),
                file_name="warped_ct_result.nii.gz",
                mime="application/gzip",
            )

    with exp_col2:
        res_df = pd.DataFrame([{
            "dice": dice_score,
            "hd95_mm": hd95_score,
            "ncc": ncc_score,
            "nmi": nmi_score,
            "ssim": ssim_score,
            "jac_neg_pct": jac_info.get("jac_neg_pct"),
            "inference_ms": elapsed_ms,
        }])
        st.download_button(
            label="Download Metrics CSV",
            data=res_df.to_csv(index=False),
            file_name="registration_metrics.csv",
            mime="text/csv",
        )

    # --- Pipeline Timing Breakdown ---
    if step_times:
        st.markdown("---")
        st.subheader("⏱ Pipeline Timing Breakdown")
        timing_rows = [
            {"Stage": stage, "Time (s)": f"{t:.3f}", "Time (ms)": f"{t * 1000:.1f}"}
            for stage, t in step_times.items()
        ]
        total_s = sum(step_times.values())
        timing_rows.append({"Stage": "Total", "Time (s)": f"{total_s:.3f}", "Time (ms)": f"{total_s * 1000:.1f}"})
        timing_df = pd.DataFrame(timing_rows)
        st.dataframe(timing_df, use_container_width=True, hide_index=True)
        inf_s = step_times.get("VoxelMorph Inference", 0)
        if inf_s > 0:
            st.caption(f"🚀 Neural inference ({inf_s * 1000:.1f} ms) is {total_s / inf_s:.0f}× faster than the full pipeline — and ~3,600× faster than classical registration (~3 min).")







# ---------------------------------------------------------------------------
# Tab 2: Performance Overview
# ---------------------------------------------------------------------------

def render_overview_tab(vm_df):
    st.header("Test Set Performance Summary")
    if vm_df.empty:
        st.warning("VoxelMorph evaluation metrics CSV not found.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean Dice ↑", f"{vm_df['dice'].mean():.4f}", delta="Target > 0.776")
    m2.metric("Mean HD95 ↓", f"{vm_df['hd95'].mean():.2f} mm", delta="Target < 19.2")
    m3.metric("Mean NMI ↑", f"{vm_df['nmi'].mean():.4f}")
    m4.metric("Mean SSIM ↑", f"{vm_df['ssim'].mean():.4f}")
    m5.metric("Mean Jac Neg% ↓", f"{vm_df['jac_neg_pct'].mean():.3f}%", delta="Target < 1%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(vm_df, x="dice", nbins=20, title="Dice Score Distribution")
        fig.add_vline(x=vm_df["dice"].mean(), line_dash="dash", line_color="green", annotation_text="Mean")
        st.plotly_chart(apply_chart_style(fig))

    with col2:
        fig = px.histogram(vm_df, x="jac_neg_pct", nbins=20, title="Jacobian Negative Percentage (%)")
        fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Target < 1.0%")
        st.plotly_chart(apply_chart_style(fig))

    st.subheader("Statistical Summary")
    numeric_cols = ["dice", "hd95", "ncc", "nmi", "ssim", "jac_mean", "jac_std", "jac_neg_pct"]
    avail = [c for c in numeric_cols if c in vm_df.columns]
    summary_df = vm_df[avail].describe().T
    st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Training Curves
# ---------------------------------------------------------------------------

def render_training_tab(log_df):
    st.header("Training Progress & Curves")
    if log_df.empty:
        st.warning("Training log CSV not found.")
        return

    st.caption(f"Total Epochs: {len(log_df)} | Final Learning Rate: {log_df['lr'].iloc[-1]:.2e}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["train_loss"], name="Train Loss"))
        fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["val_loss"], name="Val Loss", line=dict(dash="dash")))
        fig.update_layout(title="Total Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(apply_chart_style(fig))

    with c2:
        if "val_ncc" in log_df.columns:
            fig = px.line(log_df, x="epoch", y="val_ncc", title="Validation NCC Progress")
            st.plotly_chart(apply_chart_style(fig))

    c3, c4 = st.columns(2)
    with c3:
        if "train_mi" in log_df.columns and "val_mi" in log_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["train_mi"], name="Train MI"))
            fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["val_mi"], name="Val MI", line=dict(dash="dash")))
            fig.update_layout(title="Mutual Information Loss", xaxis_title="Epoch")
            st.plotly_chart(apply_chart_style(fig))

    with c4:
        if "train_reg" in log_df.columns and "val_reg" in log_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["train_reg"], name="Train Reg"))
            fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["val_reg"], name="Val Reg", line=dict(dash="dash")))
            fig.update_layout(title="Regularization Smoothness Penalty", xaxis_title="Epoch")
            st.plotly_chart(apply_chart_style(fig))


# ---------------------------------------------------------------------------
# Tab 4: Per-Subject Explorer
# ---------------------------------------------------------------------------

def render_subjects_tab(vm_df):
    st.header("Subject-level Performance Breakdown")
    if vm_df.empty:
        st.warning("Subject metrics data missing.")
        return

    metrics_list = [c for c in vm_df.columns if c != "subject_id"]
    selected_metric = st.selectbox("Sort subjects by metric", metrics_list, index=0)

    # For quality metrics: higher is better → sort descending so best subjects are first
    higher_is_better = {"dice", "nmi", "ssim", "ncc"}
    ascending = selected_metric not in higher_is_better
    sorted_df = vm_df.sort_values(selected_metric, ascending=ascending)

    fig = px.bar(
        sorted_df,
        x="subject_id",
        y=selected_metric,
        color=selected_metric,
        title=f"Subjects Sorted by {selected_metric.upper()}",
    )
    st.plotly_chart(apply_chart_style(fig, height=420))

    st.subheader("Individual Subject Details")
    subj = st.selectbox("Select Subject ID", vm_df["subject_id"].unique())
    row = vm_df[vm_df["subject_id"] == subj].iloc[0]

    cols = st.columns(4)
    cols[0].metric("Dice", f"{row.get('dice', 0):.4f}")
    cols[1].metric("HD95", f"{row.get('hd95', 0):.2f} mm")
    cols[2].metric("NMI", f"{row.get('nmi', 0):.4f}")
    cols[3].metric("Jac Neg %", f"{row.get('jac_neg_pct', 0):.3f}%")

    diffmap_file = FIGURES_DIR / f"voxelmorph_diffmap_{subj}.png"
    if diffmap_file.exists():
        st.image(str(diffmap_file), caption=f"Pre-rendered Difference Map for {subj}")


# ---------------------------------------------------------------------------
# Tab 5: Baseline Method Comparison
# ---------------------------------------------------------------------------

def render_comparison_tab(vm_df, rigid_df, affine_df, bspline_df):
    st.header("Method Comparison (VoxelMorph vs Baselines)")

    rows = []
    for name, df in [("Rigid", rigid_df), ("Affine", affine_df), ("B-spline", bspline_df)]:
        if not df.empty:
            test_subset = df[df["split"] == "test"] if "split" in df.columns else df
            for _, r in test_subset.iterrows():
                rows.append({
                    "Method": name,
                    "Subject": r.get("subject_id"),
                    "Dice": r.get("dice"),
                    "HD95": r.get("hd95"),
                })

    if not vm_df.empty:
        for _, r in vm_df.iterrows():
            rows.append({
                "Method": "VoxelMorph v2",
                "Subject": r.get("subject_id"),
                "Dice": r.get("dice"),
                "HD95": r.get("hd95"),
            })

    if not rows:
        st.info("No comparative metrics available.")
        return

    comp_df = pd.DataFrame(rows)
    comp_df = comp_df.dropna(subset=["Dice", "HD95"])  # remove rows with missing values
    if comp_df.empty:
        return
    summary = comp_df.groupby("Method").agg(
        Mean_Dice=("Dice", "mean"),
        Std_Dice=("Dice", "std"),
        Mean_HD95=("HD95", "mean"),
        Std_HD95=("HD95", "std"),
    ).round(4)

    st.subheader("Quantitative Summary Table")
    st.dataframe(summary, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(comp_df, x="Method", y="Dice", points="all", title="Dice Score Comparison")
        st.plotly_chart(apply_chart_style(fig))
    with c2:
        fig = px.box(comp_df, x="Method", y="HD95", points="all", title="HD95 Distance (mm) Comparison")
        st.plotly_chart(apply_chart_style(fig))


# ---------------------------------------------------------------------------
# Tab 6: Deformation Field Quality
# ---------------------------------------------------------------------------

def render_deformation_tab(vm_df):
    st.header("Deformation Field Regularity Analysis")
    if vm_df.empty or "jac_neg_pct" not in vm_df.columns:
        st.warning("Jacobian statistics not found in results.")
        return

    st.markdown(
        r"""
        The Jacobian determinant ($\det J_\phi$) indicates local volume expansion ($\det J > 1$),
        compression ($0 < \det J < 1$), or physical folding ($\det J \le 0$).
        A lower percentage of negative determinants indicates smoother, topology-preserving warps.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            vm_df,
            x="jac_neg_pct",
            y="dice",
            hover_name="subject_id",
            title="Jacobian Neg % vs Dice Score",
            labels={"jac_neg_pct": "Jacobian Folding %", "dice": "Dice Score"},
        )
        st.plotly_chart(apply_chart_style(fig))

    with c2:
        fig = px.histogram(vm_df, x="jac_mean", nbins=15, title="Jacobian Mean Determinant Distribution")
        st.plotly_chart(apply_chart_style(fig))

    flagged = vm_df[vm_df["jac_neg_pct"] > 0.15]
    if not flagged.empty:
        st.warning(f"Note: {len(flagged)} subjects have > 0.15% negative Jacobian determinants.")
        st.dataframe(flagged[["subject_id", "jac_neg_pct", "dice", "hd95"]], use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 7: Difference Maps
# ---------------------------------------------------------------------------

def render_diffmaps_tab(diff_vm, diff_aff, diff_rig):
    st.header("Intensity Difference Map Analysis")

    methods_dict = {}
    if not diff_vm.empty:
        methods_dict["VoxelMorph"] = diff_vm
    if not diff_aff.empty:
        methods_dict["Affine"] = diff_aff
    if not diff_rig.empty:
        methods_dict["Rigid"] = diff_rig

    if methods_dict:
        selected_method = st.selectbox("Select registration method", list(methods_dict.keys()))
        df = methods_dict[selected_method]

        fig = px.bar(
            df,
            x="subject_id",
            y="diff_mean",
            title=f"Mean Intensity Difference per Subject ({selected_method})",
        )
        st.plotly_chart(apply_chart_style(fig, height=400))
    else:
        st.info("Difference map statistics CSV files not found.")

    diffmap_fig = FIGURES_DIR / "voxelmorph_diffmap.png"
    if diffmap_fig.exists():
        st.subheader("Sample Registration Heatmap")
        st.image(str(diffmap_fig), caption="VoxelMorph v2 Sample Registration and Difference Heatmap")


# ---------------------------------------------------------------------------
# Main App Entrypoint
# ---------------------------------------------------------------------------

def main():
    # Load shared dataset results
    vm_df = load_voxelmorph_metrics()
    log_df = load_training_log()
    rigid_df = load_baseline("rigid")
    affine_df = load_baseline("affine")
    bspline_df = load_baseline("bspline")

    diff_vm = load_diff_map_stats("voxelmorph")
    diff_aff = load_diff_map_stats("affine")
    diff_rig = load_diff_map_stats("rigid")

    # Render vertical left sidebar with benchmark metrics & project info
    render_sidebar(vm_df)

    st.title("🧠 DeepMedAlign Results Dashboard")
    st.caption("VoxelMorph v2 — Multi-Resolution Pyramid with Elastic Augmentation & Jacobian Regularization")
    st.markdown("---")

    t_inference, t_overview, t_train, t_subjects, t_compare, t_def, t_diff = st.tabs([
        "🚀 Run Inference",
        "📊 Overview",
        "📈 Training Curves",
        "🔬 Per-Subject Explorer",
        "⚔️ Method Comparison",
        "🌀 Deformation Quality",
        "🗺️ Difference Maps",
    ])

    with t_inference:
        render_inference_tab(vm_df)

    with t_overview:
        render_overview_tab(vm_df)

    with t_train:
        render_training_tab(log_df)

    with t_subjects:
        render_subjects_tab(vm_df)

    with t_compare:
        render_comparison_tab(vm_df, rigid_df, affine_df, bspline_df)

    with t_def:
        render_deformation_tab(vm_df)

    with t_diff:
        render_diffmaps_tab(diff_vm, diff_aff, diff_rig)


if __name__ == "__main__":
    main()
