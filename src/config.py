"""Central configuration: paths, constants, and settings for the whole project."""
from pathlib import Path

# --- Folders (built relative to this file, so it works on any machine) ---
BASE_DIR = Path(__file__).resolve().parent.parent   # the fake-news-detector/ root
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# --- Dataset files (you'll drop these in during Step 6) ---
FAKE_CSV = RAW_DATA_DIR / "Fake.csv"
TRUE_CSV = RAW_DATA_DIR / "True.csv"
CLEAN_CSV = PROCESSED_DATA_DIR / "clean_news.csv"   # produced in Phase 1

# --- Label convention ---
# We treat FAKE as the "positive" class (the thing we're trying to catch),
# so precision/recall read naturally: "of what we flagged fake, how much really was?"
LABEL_FAKE = 1
LABEL_REAL = 0

# --- Reproducibility ---
RANDOM_SEED = 42
TEST_SIZE = 0.2   # 20% of data held out for testing
