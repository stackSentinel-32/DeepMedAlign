from pathlib import Path

from src.config import (
    CONFIGS,
    DATA,
    DOCS,
    FIXED_SHAPE,
    FIGURES,
    HU_CLIP_RANGE,
    LOGS,
    MODELS,
    PAPER_NOTES,
    PROCESSED,
    RAW,
    RESULTS,
    ROOT,
    SCRIPTS,
    SYNTHRAD,
    TESTS,
    VOXEL_SPACING,
    ensure_project_dirs,
)


# Ensure directories exist for tests; directory creation is explicit now
ensure_project_dirs()


def test_all_dirs_created():
    for directory in (
        DATA,
        RAW,
        PROCESSED,
        SYNTHRAD,
        MODELS,
        RESULTS,
        FIGURES,
        LOGS,
        DOCS,
        PAPER_NOTES,
        CONFIGS,
        SCRIPTS,
        TESTS,
    ):
        assert directory.exists()
        assert directory.is_dir()


def test_constants_correct_types():
    assert isinstance(ROOT, Path)
    assert isinstance(SYNTHRAD, Path)
    assert isinstance(VOXEL_SPACING, tuple)
    assert isinstance(FIXED_SHAPE, tuple)
    assert isinstance(HU_CLIP_RANGE, tuple)


def test_voxel_spacing_values():
    assert len(VOXEL_SPACING) == 3
    assert all(value > 0 for value in VOXEL_SPACING)


def test_fixed_shape_values():
    assert len(FIXED_SHAPE) == 3
    assert all(isinstance(value, int) and value > 0 for value in FIXED_SHAPE)


def test_hu_clip_range():
    low, high = HU_CLIP_RANGE
    assert low < high


def test_src_package_importable():
    import src  # noqa: F401
