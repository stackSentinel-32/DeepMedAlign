from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SYNTHRAD = RAW / "synthrad"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"
PAPER_NOTES = ROOT / "paper_notes"
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"

VOXEL_SPACING = (1.0, 1.0, 1.0)
FIXED_SHAPE = (192, 192, 192)
HU_CLIP_RANGE = (-1024, 3071)


def ensure_project_dirs():
    """Create project directories. Call this from setup/CLI entrypoints.

    This is intentionally not executed at import time to avoid surprising
    side-effects for consumers that merely import `src.config`.
    """
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
        directory.mkdir(parents=True, exist_ok=True)
