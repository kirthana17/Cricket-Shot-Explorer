from pathlib import Path


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

DATASET_PATH = DATA_DIR / "shots.csv"


# Model paths
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"


# Report paths
REPORTS_DIR = ROOT_DIR / "reports"


# Reproducibility
RANDOM_STATE = 42


# Machine learning features
FEATURES = [
    "bat_speed",
    "impact_height",
    "ball_length",
    "ball_line",
    "timing",
    "front_foot",
]


# Target variable
TARGET = "shot_type"


# Shot classes
SHOT_CLASSES = [
    "cover_drive",
    "straight_drive",
    "flick",
    "pull",
    "cut",
    "defensive",
]