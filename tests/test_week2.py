"""
test_week2.py
-------------
Week 2 test suite for the classical registration pipeline.
Tests are structured so they skip cleanly when data is absent
(e.g. on a fresh clone without preprocessed files).

Run:
  pytest tests/test_week2.py -v
"""

import pytest
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg():
    """Import config once for the whole session."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src import config as c
    return c


@pytest.fixture(scope="session")
def manifest(cfg):
    """Return the manifest DataFrame if available, else skip."""
    import pandas as pd
    p = cfg.DATA_RAW / "manifest.csv"
    if not p.exists():
        pytest.skip("No manifest yet — run build_manifest.py first.")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def manifest_registered(cfg):
    """Return manifest_registered.csv if available."""
    import pandas as pd
    p = cfg.DATA_RAW / "manifest_registered.csv"
    if not p.exists():
        pytest.skip("No manifest_registered.csv — run run_classical.py first.")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def baseline_affine(cfg):
    """Return affine baseline metrics CSV if available."""
    import pandas as pd
    p = cfg.RESULTS / "baseline_metrics_affine.csv"
    if not p.exists():
        pytest.skip("No affine metrics — run compute_baseline_metrics.py first.")
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# src/metrics.py unit tests (no data required)
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    """Pure unit tests for metric functions — no real images required."""

    def test_dice_perfect_overlap(self):
        from src.metrics import dice_coefficient
        mask = np.ones((10, 10, 10), dtype=bool)
        assert dice_coefficient(mask, mask) == pytest.approx(1.0)

    def test_dice_no_overlap(self):
        from src.metrics import dice_coefficient
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)
        a[:5, :, :] = True
        b[5:, :, :] = True
        assert dice_coefficient(a, b) == pytest.approx(0.0)

    def test_dice_both_empty(self):
        from src.metrics import dice_coefficient
        empty = np.zeros((5, 5, 5), dtype=bool)
        assert dice_coefficient(empty, empty) == pytest.approx(1.0)

    def test_dice_partial_overlap(self):
        from src.metrics import dice_coefficient
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)
        a[:8, :, :] = True   # 800 voxels
        b[:4, :, :] = True   # 400 voxels; overlap = 400
        expected = 2 * 400 / (800 + 400)
        assert dice_coefficient(a, b) == pytest.approx(expected, abs=1e-4)

    def test_hausdorff95_identical_masks(self):
        from src.metrics import hausdorff95
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:15, 5:15, 5:15] = True
        assert hausdorff95(mask, mask) == pytest.approx(0.0, abs=1e-3)

    def test_hausdorff95_empty_mask_returns_inf(self):
        from src.metrics import hausdorff95
        empty = np.zeros((10, 10, 10), dtype=bool)
        full  = np.ones((10, 10, 10),  dtype=bool)
        assert hausdorff95(empty, full) == float("inf")

    def test_ncc_identical_images(self):
        from src.metrics import normalised_cross_correlation
        arr = np.random.rand(10, 10, 10).astype("float32")
        assert normalised_cross_correlation(arr, arr) == pytest.approx(1.0, abs=1e-5)

    def test_ncc_range(self):
        from src.metrics import normalised_cross_correlation
        a = np.random.rand(10, 10, 10).astype("float32")
        b = np.random.rand(10, 10, 10).astype("float32")
        val = normalised_cross_correlation(a, b)
        assert -1.0 <= val <= 1.0

    def test_jacobian_stats_uniform_flow(self):
        from src.metrics import jacobian_stats
        # Zero flow -> Jac = 1 everywhere, no folding
        flow = np.zeros((3, 16, 16, 16), dtype="float32")
        stats = jacobian_stats(flow)
        assert stats["jac_neg_pct"] == pytest.approx(0.0)
        assert stats["jac_mean"] == pytest.approx(1.0, abs=1e-4)

    def test_jacobian_stats_bad_input_raises(self):
        from src.metrics import jacobian_stats
        with pytest.raises(ValueError):
            jacobian_stats(np.zeros((2, 10, 10, 10)))


# ---------------------------------------------------------------------------
# src/classical_reg.py — import and function signature tests
# ---------------------------------------------------------------------------

class TestClassicalRegImports:
    """Verify the module imports and exposes the expected API."""

    def test_module_importable(self):
        from src import classical_reg  # noqa: F401

    def test_register_rigid_callable(self):
        from src.classical_reg import register_rigid
        assert callable(register_rigid)

    def test_register_affine_callable(self):
        from src.classical_reg import register_affine
        assert callable(register_affine)

    def test_register_bspline_callable(self):
        from src.classical_reg import register_bspline
        assert callable(register_bspline)

    def test_register_full_pipeline_callable(self):
        from src.classical_reg import register_full_pipeline
        assert callable(register_full_pipeline)


# ---------------------------------------------------------------------------
# Data-dependent tests (skip if files absent)
# ---------------------------------------------------------------------------

class TestRegisteredOutputs:
    """Verify that registered output files exist and are non-trivial."""

    def test_affine_outputs_exist(self, cfg, manifest):
        """At least one affine-warped CT must exist in data/processed."""
        affines = list(cfg.DATA_PROC.glob("*/*_ct_affine.nii.gz"))
        if not affines:
            pytest.skip("No affine outputs found — run run_classical.py first.")
        assert len(affines) > 0

    def test_rigid_outputs_exist(self, cfg, manifest):
        """At least one rigid-warped CT must exist in data/processed."""
        rigids = list(cfg.DATA_PROC.glob("*/*_ct_rigid.nii.gz"))
        if not rigids:
            pytest.skip("No rigid outputs found — run run_classical.py first.")
        assert len(rigids) > 0

    def test_manifest_registered_has_reg_status(self, manifest_registered):
        """manifest_registered.csv must contain a reg_status column."""
        assert "reg_status" in manifest_registered.columns

    def test_manifest_registered_no_subject_duplicates(self, manifest_registered):
        """Each subject must appear exactly once in manifest_registered."""
        counts = manifest_registered["subject_id"].value_counts()
        dupes  = counts[counts > 1]
        assert dupes.empty, f"Duplicate subjects: {dupes.index.tolist()}"


class TestBaselineMetrics:
    """Verify the affine baseline CSV is sane."""

    def test_affine_metrics_has_required_columns(self, baseline_affine):
        for col in ("subject_id", "status", "method", "ncc"):
            assert col in baseline_affine.columns, f"Missing column: {col}"

    def test_affine_metrics_ncc_range(self, baseline_affine):
        ok  = baseline_affine[baseline_affine["status"] == "ok"]
        ncc = ok["ncc"].dropna()
        if ncc.empty:
            pytest.skip("No NCC values computed.")
        assert (ncc >= -1.0).all(), "NCC below -1 is impossible."
        assert (ncc <=  1.0).all(), "NCC above  1 is impossible."

    def test_affine_dice_non_negative(self, baseline_affine):
        ok   = baseline_affine[baseline_affine["status"] == "ok"]
        dice = ok["dice"].dropna()
        if dice.empty:
            pytest.skip("No Dice values computed (CT masks missing?).")
        assert (dice >= 0.0).all(), "Dice cannot be negative."
        assert (dice <= 1.0).all(), "Dice cannot exceed 1."
