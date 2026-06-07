"""Tests that R1's data pipeline outputs exist and are valid."""
import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_RAW


def manifest_exists():
    """Helper: returns True if any manifest exists."""
    return any([
        (DATA_RAW / "manifest.csv").exists(),
        (DATA_RAW / "manifest_v2.csv").exists(),
        (DATA_RAW / "manifest_final.csv").exists(),
    ])


@pytest.mark.skipif(not manifest_exists(),
                    reason="No manifest yet - run build_manifest.py first")
class TestManifest:

    def get_manifest(self):
        for name in ["manifest_final.csv", "manifest_v2.csv", "manifest.csv"]:
            p = DATA_RAW / name
            if p.exists():
                return pd.read_csv(p)
        pytest.skip("No manifest found")

    def test_manifest_has_required_columns(self):
        df = self.get_manifest()
        required = ["subject_id", "mr", "ct", "split"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_manifest_has_three_splits(self):
        df = self.get_manifest()
        splits = set(df["split"].unique())
        assert splits == {"train", "val", "test"}, f"Expected train/val/test, got {splits}"

    def test_no_subject_in_multiple_splits(self):
        """Critical: no subject_id appears in more than one split."""
        df = self.get_manifest()
        for sid in df["subject_id"]:
            count = len(df[df["subject_id"] == sid])
            assert count == 1, f"Subject {sid} appears in {count} splits - DATA LEAKAGE"

    def test_train_is_largest_split(self):
        df = self.get_manifest()
        counts = df["split"].value_counts()
        assert counts["train"] > counts["val"]
        assert counts["train"] > counts["test"]

    def test_all_mr_paths_are_strings(self):
        df = self.get_manifest()
        assert all(isinstance(x, str) for x in df["mr"])
        assert not df["mr"].isnull().any()

    def test_at_least_5_subjects(self):
        df = self.get_manifest()
        assert len(df) >= 5, f"Only {len(df)} subjects - expected at least 5"


@pytest.mark.skipif(not (DATA_RAW / "shape_report.csv").exists(),
                    reason="No shape report yet - run validate_data.py first")
class TestShapeReport:

    def test_shape_report_has_required_columns(self):
        df = pd.read_csv(DATA_RAW / "shape_report.csv")
        required = ["subject_id", "mr_shape", "ct_shape",
                    "ct_min_hu", "ct_max_hu", "flags", "status"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_load_errors(self):
        df = pd.read_csv(DATA_RAW / "shape_report.csv")
        load_errors = df[df["flags"].str.contains("LOAD_ERROR", na=False)]
        assert len(load_errors) == 0, \
            f"{len(load_errors)} subjects have load errors:\n{load_errors[['subject_id','flags']]}"

    def test_ct_has_negative_hu(self):
        """CT must contain negative HU values (air = -1000)."""
        df = pd.read_csv(DATA_RAW / "shape_report.csv")
        if "ct_min_hu" in df.columns:
            bad = df[df["ct_min_hu"] > -100]
            assert len(bad) == 0, \
                f"{len(bad)} subjects have suspicious CT HU min (> -100)"

    def test_ct_has_bone_hu(self):
        """CT bone should reach at least 200 HU."""
        df = pd.read_csv(DATA_RAW / "shape_report.csv")
        if "ct_max_hu" in df.columns:
            bad = df[df["ct_max_hu"] < 200]
            assert len(bad) == 0, \
                f"{len(bad)} subjects have suspicious CT HU max (< 200)"
