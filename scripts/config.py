from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"
HIDDEN_DIALOGUE_PATH = DATA_DIR / "hiddentest_dialogue.csv"
HIDDEN_TOPIC_PATH = DATA_DIR / "hiddentest_topic.csv"
BASELINE_PATH = DATA_DIR / "baseline.csv"
RESULTS_PATH = RESULTS_DIR / "results.csv"

TEST_CLEAN_PATH = DATA_DIR / "test_clean.csv"
LORA_DIR = MODELS_DIR / "lora"

SEED = 42