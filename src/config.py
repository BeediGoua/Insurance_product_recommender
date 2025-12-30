from pathlib import Path

# Paths
# Assuming src/config.py is 1 level deep in src, so parent.parent is project root
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"

# Baseline Params
BASELINE_VERSION = "baseline_v0"
OUT_DIR = ARTIFACTS_DIR / BASELINE_VERSION
MIN_BASKET_SIZE_EVAL = 2
EVAL_SEED = 42
