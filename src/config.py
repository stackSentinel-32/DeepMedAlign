from pathlib import Path

# Root paths
ROOT        = Path(__file__).parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
RESULTS     = ROOT / "results"
LOGS        = ROOT / "logs"
DOCS        = ROOT / "docs"
SCRIPTS     = ROOT / "scripts"

# Dataset paths
SYNTHRAD    = DATA_RAW / "synthrad" / "brain"
MANIFEST    = DATA_RAW / "manifest.csv"
MANIFEST_V2 = DATA_RAW / "manifest_v2.csv"
MANIFEST_P  = DATA_RAW / "manifest_processed.csv"
MANIFEST_F  = DATA_RAW / "manifest_final.csv"
SHAPE_RPT   = DATA_RAW / "shape_report.csv"

# Preprocessing constants
VOXEL_SPACING   = (1.0, 1.0, 1.0)
FIXED_SHAPE     = (160, 192, 160)
CT_HU_CLIP      = (-1000.0, 1000.0)
CT_HU_BRAIN     = (-15.0, 80.0)

# Training defaults
LR              = 1e-4
EPOCHS          = 1500
BATCH_SIZE      = 1
LAMBDA_SMOOTH   = 1.0
LAMBDA_MI       = 1.0

# Splits
RANDOM_SEED     = 42
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.10
TEST_FRAC       = 0.20

# Auto-create all output dirs on import
for _d in [DATA_RAW, DATA_PROC, MODELS_DIR, RESULTS, LOGS, DOCS,
           RESULTS / "figures"]:
    _d.mkdir(parents=True, exist_ok=True)
