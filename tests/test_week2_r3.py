"""
test_week2_r3.py
----------------
Tests for R3 Week 2 visualisation and difference-map functions.
All tests use synthetic numpy arrays — no real dataset required.

Run:
  pytest tests/test_week2_r3.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.difference_maps import (
    match_intensity_ranges,
    compute_difference_map,
    normalize_diff_by_local_std,
    difference_map_stats,
)
from scripts.checkerboard_qc import checkerboard_2d, norm_display


# ---------------------------------------------------------------------------
# match_intensity_ranges
# ---------------------------------------------------------------------------

class TestMatchIntensityRanges:

    def test_returns_two_arrays_same_shape(self):
        mr = np.random.randn(20, 20, 20).astype("float32")
        ct = np.random.rand(20, 20, 20).astype("float32")
        mr_s, ct_s = match_intensity_ranges(mr, ct)
        assert mr_s.shape == mr.shape
        assert ct_s.shape == ct.shape

    def test_output_mostly_in_zero_one(self):
        """Clipped output should lie within [0, 1]."""
        mr = np.random.randn(20, 20, 20).astype("float32") * 10
        ct = np.random.rand(20, 20, 20).astype("float32")
        mr_s, ct_s = match_intensity_ranges(mr, ct)
        # clip is applied, so values must be in [0, 1]
        assert mr_s.min() >= 0.0 - 1e-6
        assert mr_s.max() <= 1.0 + 1e-6

    def test_with_brain_mask(self):
        """Should work with a binary mask and not raise."""
        mr   = np.random.randn(20, 20, 20).astype("float32")
        ct   = np.random.rand(20, 20, 20).astype("float32")
        mask = np.zeros((20, 20, 20), dtype="float32")
        mask[5:15, 5:15, 5:15] = 1.0
        mr_s, ct_s = match_intensity_ranges(mr, ct, mask)
        assert mr_s.shape == mr.shape
        assert ct_s.shape == ct.shape

    def test_empty_mask_does_not_crash(self):
        """All-zero mask should return without error."""
        mr   = np.random.rand(10, 10, 10).astype("float32")
        ct   = np.random.rand(10, 10, 10).astype("float32")
        mask = np.zeros((10, 10, 10), dtype="float32")
        mr_s, ct_s = match_intensity_ranges(mr, ct, mask)
        assert mr_s.shape == mr.shape


# ---------------------------------------------------------------------------
# compute_difference_map
# ---------------------------------------------------------------------------

class TestComputeDifferenceMap:

    def test_identical_images_near_zero(self):
        """Identical inputs → absolute diff should be ≈ 0."""
        a    = np.random.rand(20, 20, 20).astype("float32")
        diff = compute_difference_map(a, a.copy())
        assert diff.mean() < 0.02

    def test_shape_mismatch_raises_value_error(self):
        a = np.random.rand(20, 20, 20).astype("float32")
        b = np.random.rand(10, 10, 10).astype("float32")
        with pytest.raises(ValueError):
            compute_difference_map(a, b)

    def test_mask_zeroes_outside_region(self):
        """Voxels outside the brain mask must be exactly 0."""
        a    = np.random.rand(20, 20, 20).astype("float32")
        b    = np.random.rand(20, 20, 20).astype("float32")
        mask = np.zeros((20, 20, 20), dtype="float32")
        mask[5:15, 5:15, 5:15] = 1.0
        diff = compute_difference_map(a, b, mask)
        assert np.all(diff[mask == 0] == 0.0)

    def test_absolute_diff_non_negative(self):
        a = np.random.rand(15, 15, 15).astype("float32")
        b = np.random.rand(15, 15, 15).astype("float32")
        diff = compute_difference_map(a, b, method="absolute")
        assert diff.min() >= 0.0

    def test_signed_method_can_be_negative(self):
        """Signed diff should produce negative values when MR < CT after rescaling."""
        rng = np.random.default_rng(42)
        # MR: low values (0.0 – 0.1);  CT: high values (0.9 – 1.0)
        a = rng.uniform(0.0, 0.1, (20, 20, 20)).astype("float32")
        b = rng.uniform(0.9, 1.0, (20, 20, 20)).astype("float32")
        diff = compute_difference_map(a, b, method="signed")
        # After rescaling, MR should be lower than CT → signed diff < 0
        assert diff.min() < 0.0

    def test_unknown_method_raises(self):
        a = np.random.rand(5, 5, 5).astype("float32")
        with pytest.raises(ValueError):
            compute_difference_map(a, a.copy(), method="invalid")


# ---------------------------------------------------------------------------
# normalize_diff_by_local_std
# ---------------------------------------------------------------------------

class TestNormalizeDiffByLocalStd:

    def test_output_std_near_one_with_mask(self):
        """Within-mask std of normalised output should be ≈ 1."""
        diff = np.random.randn(20, 20, 20).astype("float32") * 5
        mask = np.ones((20, 20, 20), dtype="float32")
        result = normalize_diff_by_local_std(diff, mask)
        assert abs(result[mask > 0].std() - 1.0) < 0.1

    def test_empty_mask_returns_unchanged_shape(self):
        """All-zero mask should return an array of the same shape."""
        diff = np.random.rand(10, 10, 10).astype("float32")
        mask = np.zeros((10, 10, 10), dtype="float32")
        result = normalize_diff_by_local_std(diff, mask)
        assert result.shape == diff.shape

    def test_no_mask_uses_full_array(self):
        """Without a mask the function should still return a valid array."""
        diff   = np.random.rand(15, 15, 15).astype("float32")
        result = normalize_diff_by_local_std(diff)
        assert result.shape == diff.shape


# ---------------------------------------------------------------------------
# difference_map_stats
# ---------------------------------------------------------------------------

class TestDifferenceMapStats:

    def test_returns_all_required_keys(self):
        diff  = np.random.rand(20, 20, 20).astype("float32")
        stats = difference_map_stats(diff)
        for key in ("diff_mean", "diff_std", "diff_median", "diff_p95", "diff_max"):
            assert key in stats, f"Missing key: {key}"

    def test_all_zeros_returns_zero_stats(self):
        diff  = np.zeros((10, 10, 10), dtype="float32")
        stats = difference_map_stats(diff)
        assert stats["diff_mean"] == 0.0
        assert stats["diff_max"]  == 0.0

    def test_values_are_float(self):
        diff  = np.random.rand(10, 10, 10).astype("float32")
        stats = difference_map_stats(diff)
        for v in stats.values():
            assert isinstance(v, float)

    def test_max_geq_p95(self):
        diff  = np.random.rand(20, 20, 20).astype("float32")
        stats = difference_map_stats(diff)
        assert stats["diff_max"] >= stats["diff_p95"]


# ---------------------------------------------------------------------------
# checkerboard_2d
# ---------------------------------------------------------------------------

class TestCheckerboard2D:

    def test_output_shape_matches_input(self):
        a  = np.random.rand(40, 40).astype("float32")
        b  = np.random.rand(40, 40).astype("float32")
        cb = checkerboard_2d(a, b, tile=10)
        assert cb.shape == a.shape

    def test_first_tile_from_a(self):
        """Top-left tile (block index 0,0 → even → from img_a)."""
        a  = np.ones((40, 40),  dtype="float32")
        b  = np.zeros((40, 40), dtype="float32")
        cb = checkerboard_2d(a, b, tile=10)
        assert cb[0, 0] == pytest.approx(1.0)

    def test_second_tile_from_b(self):
        """Tile at column offset 10 (block 0,1 → odd → from img_b)."""
        a  = np.ones((40, 40),  dtype="float32")
        b  = np.zeros((40, 40), dtype="float32")
        cb = checkerboard_2d(a, b, tile=10)
        assert cb[0, 10] == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        a = np.random.rand(30, 30).astype("float32")
        b = np.random.rand(20, 20).astype("float32")
        with pytest.raises(ValueError):
            checkerboard_2d(a, b)


# ---------------------------------------------------------------------------
# norm_display
# ---------------------------------------------------------------------------

class TestNormDisplay:

    def test_output_in_zero_one(self):
        a      = np.random.rand(30, 30).astype("float32") * 1000
        result = norm_display(a)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_all_zeros_handled_without_crash(self):
        a      = np.zeros((10, 10), dtype="float32")
        result = norm_display(a)
        assert result.shape == a.shape

    def test_single_value_array(self):
        """Array with a single unique non-zero value should not crash."""
        a      = np.full((5, 5), 7.0, dtype="float32")
        result = norm_display(a)
        assert result.shape == a.shape
