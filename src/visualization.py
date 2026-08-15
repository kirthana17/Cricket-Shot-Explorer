from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "shots.csv"
REPORTS_PATH = ROOT / "reports"

FEATURES = [
    "bat_speed",
    "impact_height",
    "ball_length",
    "ball_line",
    "timing",
    "front_foot",
]


def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python src/data.py` first."
        )

    return pd.read_csv(
        DATA_PATH
    )


def create_visualizations():

    df = load_data()

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Class distribution

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.countplot(
        data=df,
        x="shot_type",
        ax=ax,
    )

    ax.set_title(
        "Shot Class Distribution"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    plt.tight_layout()

    fig.savefig(
        REPORTS_PATH
        / "class_distribution.png",
        dpi=150,
    )

    plt.close(fig)

    # Feature distributions

    for feature in FEATURES:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.histplot(
            data=df,
            x=feature,
            hue="shot_type",
            kde=True,
            common_norm=False,
            ax=ax,
        )

        ax.set_title(
            f"{feature} Distribution"
        )

        plt.tight_layout()

        fig.savefig(
            REPORTS_PATH
            / f"{feature}_distribution.png",
            dpi=150,
        )

        plt.close(fig)

    # Correlation matrix

    correlation = df[
        FEATURES
    ].corr()

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    fig.savefig(
        REPORTS_PATH
        / "correlation_matrix.png",
        dpi=150,
    )

    plt.close(fig)


if __name__ == "__main__":
    create_visualizations()