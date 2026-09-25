import os
from pathlib import Path

# PROJECT_ROOT is f:\Cognivision AI
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATASET_TRAIN_PARQUET = PROJECT_ROOT / "train-00000-of-00001.parquet"
DATASET_TEST_PARQUET = PROJECT_ROOT / "test-00000-of-00001.parquet"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_CACHE_FILE = CACHE_DIR / "train_cache.pt"
TEST_CACHE_FILE = CACHE_DIR / "test_cache.pt"
TRAIN_META_FILE = CACHE_DIR / "train_metadata.json"
TEST_META_FILE = CACHE_DIR / "test_metadata.json"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"
LATEST_MODEL_PATH = MODELS_DIR / "latest_model.pt"
METRICS_FILE = ARTIFACTS_DIR / "evaluation_metrics.json"
HISTORY_FILE = ARTIFACTS_DIR / "training_history.json"

def ensure_directories():
    """Ensure all required directories exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
