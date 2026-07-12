from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent

# Data hierarchy
DATA        = ROOT / "data"
RAW         = DATA / "raw"
PROCESSED   = DATA / "processed"

# Aliases used in R1 scripts
DATA_RAW    = RAW
DATA_PROC   = PROCESSED

# Dataset paths
SYNTHRAD    = RAW / "synthrad" / "brain"
MANIFEST    = RAW / "manifest.csv"
MANIFEST_V2 = RAW / "manifest_v2.csv"
MANIFEST_P  = RAW / "manifest_processed.csv"
MANIFEST_F  = RAW / "manifest_final.csv"
SHAPE_RPT   = RAW / "shape_report.csv"

# Other top-level directories
MODELS      = ROOT / "models"
MODELS_DIR  = MODELS          # alias
RESULTS     = ROOT / "results"
FIGURES     = RESULTS / "figures"
LOGS        = ROOT / "logs"
DOCS        = ROOT / "docs"
PAPER_NOTES = ROOT / "paper_notes"
CONFIGS     = ROOT / "configs"
SCRIPTS     = ROOT / "scripts"
TESTS       = ROOT / "tests"

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------
VOXEL_SPACING   = (1.0, 1.0, 1.0)
FIXED_SHAPE     = (160, 192, 160)
CT_HU_CLIP      = (-1000.0, 1000.0)
HU_CLIP_RANGE   = CT_HU_CLIP       # alias used in test_config.py
CT_HU_BRAIN     = (-15.0, 80.0)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
LR              = 3e-4
EPOCHS          = 1500
BATCH_SIZE      = 1
LAMBDA_SMOOTH   = 0.5
LAMBDA_MI       = 1.0

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
RANDOM_SEED     = 42
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.10
TEST_FRAC       = 0.20


# ---------------------------------------------------------------------------
# Directory creation helpers
# ---------------------------------------------------------------------------

def ensure_project_dirs() -> None:
    """Create all standard project directories.

    Call this from setup scripts or CLI entrypoints.
    Intentionally NOT executed at import time to avoid side-effects
    for modules that merely import ``src.config``.
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
