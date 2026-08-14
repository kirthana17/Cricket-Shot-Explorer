from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "shots.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

FEATURES = [
    "bat_speed",
    "impact_height",
    "ball_length",
    "ball_line",
    "timing",
    "front_foot",
]

CLASSES = {
    "cover_drive",
    "straight_drive",
    "flick",
    "pull",
    "cut",
    "defensive",
}


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_dataset_columns():
    df = pd.read_csv(DATA_PATH)

    assert list(df.columns) == (
        FEATURES + ["shot_type"]
    )


def test_dataset_size():
    df = pd.read_csv(DATA_PATH)

    assert len(df) == 5000


def test_all_classes_present():
    df = pd.read_csv(DATA_PATH)

    assert set(
        df["shot_type"].unique()
    ) == CLASSES


def test_no_missing_values():
    df = pd.read_csv(DATA_PATH)

    assert not df.isnull().any().any()


def test_feature_ranges():
    df = pd.read_csv(DATA_PATH)

    assert df["bat_speed"].between(
        20,
        150,
    ).all()

    assert df["impact_height"].between(
        0,
        180,
    ).all()

    assert df["ball_length"].between(
        0.5,
        11,
    ).all()

    assert df["ball_line"].between(
        -120,
        160,
    ).all()

    assert df["timing"].between(
        0,
        1,
    ).all()

    assert df["front_foot"].isin(
        [0, 1]
    ).all()


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_prediction():
    artifact = joblib.load(
        MODEL_PATH
    )

    model = artifact["model"]

    sample = pd.DataFrame(
        [
            {
                "bat_speed": 95,
                "impact_height": 35,
                "ball_length": 3.2,
                "ball_line": 55,
                "timing": 0.75,
                "front_foot": 1,
            }
        ]
    )

    prediction = model.predict(
        sample
    )

    assert len(prediction) == 1

    assert prediction[0] in CLASSES