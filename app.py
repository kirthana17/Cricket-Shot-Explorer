from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "shots.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cricket Shot Explorer",
    page_icon="🏏",
    layout="wide",
)


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Run `python src/data` first."
    )
    st.stop()


if not MODEL_PATH.exists():
    st.error(
        "Model not found. Run `python src/models` first."
    )
    st.stop()


if not METRICS_PATH.exists():
    st.error(
        "Metrics not found. Run `python src/models` first."
    )
    st.stop()


# ============================================================
# LOAD PROJECT ARTIFACTS
# ============================================================

df = load_dataset()

artifact = load_model()

metrics = load_metrics()

model = artifact["model"]
model_name = artifact["model_name"]
features = artifact["features"]
labels = artifact["labels"]


# ============================================================
# HEADER
# ============================================================

st.title("🏏 Cricket Shot Explorer")

st.write(
    "Predict a cricket shot using machine learning."
)

st.caption(
    "This project currently uses synthetic structured data."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Project Information")

st.sidebar.metric(
    "Dataset Rows",
    f"{len(df):,}",
)

st.sidebar.metric(
    "Features",
    len(features),
)

st.sidebar.metric(
    "Shot Classes",
    len(labels),
)

st.sidebar.write(
    f"**Model:** {model_name.replace('_', ' ').title()}"
)


# ============================================================
# TABS
# ============================================================

prediction_tab, eda_tab, performance_tab = st.tabs(
    [
        "🎯 Prediction",
        "📊 EDA",
        "📈 Model Performance",
    ]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with prediction_tab:

    st.header("Shot Prediction")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT INPUTS
    # --------------------------------------------------------

    with col1:

        bat_speed = st.slider(
            "Bat Speed (km/h)",
            min_value=20.0,
            max_value=150.0,
            value=90.0,
            step=1.0,
        )

        impact_height = st.slider(
            "Impact Height (cm)",
            min_value=0.0,
            max_value=180.0,
            value=40.0,
            step=1.0,
        )

        ball_length = st.slider(
            "Ball Length (m)",
            min_value=0.5,
            max_value=11.0,
            value=4.0,
            step=0.1,
        )

    # --------------------------------------------------------
    # RIGHT INPUTS
    # --------------------------------------------------------

    with col2:

        ball_line = st.slider(
            "Ball Line (cm)",
            min_value=-120.0,
            max_value=160.0,
            value=20.0,
            step=1.0,
            help=(
                "Negative values represent the leg side; "
                "positive values represent the off side."
            ),
        )

        timing = st.slider(
            "Timing",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.01,
            help="0 = poor timing, 1 = excellent timing.",
        )

        front_foot = st.checkbox(
            "Front Foot",
            value=True,
        )

    # --------------------------------------------------------
    # CREATE INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "bat_speed": bat_speed,
                "impact_height": impact_height,
                "ball_length": ball_length,
                "ball_line": ball_line,
                "timing": timing,
                "front_foot": int(front_foot),
            }
        ]
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🏏 Predict Shot",
        type="primary",
        use_container_width=True,
    ):

        prediction = model.predict(
            input_data[features]
        )[0]

        probabilities = model.predict_proba(
            input_data[features]
        )[0]

        confidence = probabilities.max()

        st.success(
            "Predicted Shot: "
            f"**{prediction.replace('_', ' ').title()}**"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.1%}",
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame(
            {
                "Shot": model.classes_,
                "Probability": probabilities,
            }
        ).sort_values(
            "Probability",
            ascending=False,
        )

        probability_df["Shot"] = (
            probability_df["Shot"]
            .str.replace("_", " ")
            .str.title()
        )

        st.bar_chart(
            probability_df.set_index(
                "Shot"
            )["Probability"],
            height=280,
        )

        # ----------------------------------------------------
        # SYNTHETIC COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "Comparison with Synthetic Examples"
        )

        predicted_rows = df[
            df["shot_type"] == prediction
        ]

        if not predicted_rows.empty:

            comparison_features = [
                "bat_speed",
                "impact_height",
                "ball_length",
                "ball_line",
                "timing",
            ]

            comparison = pd.DataFrame(
                {
                    "Feature": comparison_features,
                    "Your Input": [
                        input_data.iloc[0][feature]
                        for feature in comparison_features
                    ],
                    "Synthetic Mean": [
                        predicted_rows[feature].mean()
                        for feature in comparison_features
                    ],
                }
            )

            comparison["Difference"] = (
                comparison["Your Input"]
                - comparison["Synthetic Mean"]
            )

            st.dataframe(
                comparison.round(3),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# TAB 2 — EDA
# ============================================================

with eda_tab:

    st.header("Exploratory Data Analysis")

    st.caption(
        "These plots describe the synthetic dataset and should "
        "not be interpreted as real-world cricket biomechanics."
    )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Rows",
            f"{len(df):,}",
        )

    with metric2:
        st.metric(
            "Features",
            len(features),
        )

    with metric3:
        st.metric(
            "Shot Classes",
            df["shot_type"].nunique(),
        )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Shot Class Distribution")

    class_counts = (
        df["shot_type"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        class_counts,
        height=260,
    )

    # --------------------------------------------------------
    # FEATURE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Select Feature",
        [
            "bat_speed",
            "impact_height",
            "ball_length",
            "ball_line",
            "timing",
            "front_foot",
        ],
    )

    # Smaller figure
    fig, ax = plt.subplots(
        figsize=(7, 3.8),
        dpi=110,
    )

    sns.histplot(
        data=df,
        x=selected_feature,
        hue="shot_type",
        kde=True,
        common_norm=False,
        element="step",
        alpha=0.30,
        ax=ax,
    )

    ax.set_title(
        f"{selected_feature.replace('_', ' ').title()} "
        "by Shot Type"
    )

    ax.set_xlabel(
        selected_feature.replace(
            "_",
            " ",
        ).title()
    )

    ax.set_ylabel("Count")

    plt.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader("Feature Correlation")

    correlation = df[
        features
    ].corr()

    fig, ax = plt.subplots(
        figsize=(6.5, 4.8),
        dpi=110,
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        annot_kws={
            "size": 8
        },
        ax=ax,
    )

    ax.set_title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with performance_tab:

    st.header("Model Performance")

    selected = metrics[
        "models"
    ][model_name]

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Selected Model",
            model_name.replace(
                "_",
                " ",
            ).title(),
        )

    with metric2:

        st.metric(
            "Test Accuracy",
            f"{selected['accuracy']:.1%}",
        )

    with metric3:

        st.metric(
            "CV Accuracy",
            f"{selected['cv_mean_accuracy']:.1%}",
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    comparison = []

    for name, result in metrics[
        "models"
    ].items():

        comparison.append(
            {
                "Model": name.replace(
                    "_",
                    " ",
                ).title(),

                "Test Accuracy": result[
                    "accuracy"
                ],

                "CV Accuracy": result[
                    "cv_mean_accuracy"
                ],

                "CV Std": result[
                    "cv_std"
                ],

                "Macro F1": result[
                    "macro_f1"
                ],
            }
        )

    comparison_df = pd.DataFrame(
        comparison
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "Test Accuracy": "{:.1%}",
                "CV Accuracy": "{:.1%}",
                "CV Std": "{:.2%}",
                "Macro F1": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        selected["confusion_matrix"],
        index=labels,
        columns=labels,
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 4.8),
        dpi=110,
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        annot_kws={
            "size": 8
        },
        xticklabels=[
            label.replace("_", " ")
            for label in labels
        ],
        yticklabels=[
            label.replace("_", " ")
            for label in labels
        ],
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        "Shot Classification Confusion Matrix"
    )

    plt.xticks(
        rotation=35,
        ha="right",
        fontsize=8,
    )

    plt.yticks(
        rotation=0,
        fontsize=8,
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Feature Importance"
    )

    st.caption(
        "Feature importance is calculated from the Random Forest "
        "for interpretability, even when Logistic Regression is "
        "selected as the final model."
    )

    importance = (
        pd.Series(
            metrics[
                "feature_importances"
            ]
        )
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        importance,
        height=280,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cricket Shot Explorer — End-to-End ML System"
)