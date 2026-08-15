from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "shots.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
OUTPUT_PATH = (
    ROOT
    / "reports"
    / "evaluation_results.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "bat_speed",
    "impact_height",
    "ball_length",
    "ball_line",
    "timing",
    "front_foot",
]

LABEL = "shot_type"


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python src/models.py` first."
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    X = df[FEATURES].copy()
    y = df[LABEL]

    X["front_foot"] = (
        X["front_foot"].astype(int)
    )

    # --------------------------------------------------------
    # Reproduce the same stratified test split
    # used during model development.
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------------
    # Load saved model artifact
    # --------------------------------------------------------

    artifact = joblib.load(
        MODEL_PATH
    )

    model = artifact["model"]

    model_name = artifact.get(
        "model_name",
        "unknown",
    )

    # --------------------------------------------------------
    # Predict ONLY on the held-out test set
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    labels = sorted(
        y.unique()
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    result = {
        "model": model_name,
        "dataset_rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "label": LABEL,
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "macro_f1": round(
            float(
                report["macro avg"][
                    "f1-score"
                ]
            ),
            4,
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        f"Evaluation model: {model_name}"
    )

    print(
        f"Evaluation rows: {len(X_test)}"
    )

    print(
        f"Evaluation accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Evaluation macro F1: "
        f"{report['macro avg']['f1-score']:.4f}"
    )

    print(
        f"Evaluation results saved to: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()