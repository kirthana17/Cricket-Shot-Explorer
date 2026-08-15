import pandas as pd


FEATURES = [
    "bat_speed",
    "impact_height",
    "ball_length",
    "ball_line",
    "timing",
    "front_foot",
]

LABEL = "shot_type"


def validate_features(df):
    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )


def prepare_features(df):
    validate_features(df)

    X = df[FEATURES].copy()

    X["front_foot"] = (
        X["front_foot"].astype(int)
    )

    return X


def create_prediction_input(
    bat_speed,
    impact_height,
    ball_length,
    ball_line,
    timing,
    front_foot,
):
    return pd.DataFrame(
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


def validate_prediction_input(df):
    validate_features(df)

    if not df["bat_speed"].between(
        20, 150
    ).all():
        raise ValueError(
            "bat_speed must be between 20 and 150."
        )

    if not df["impact_height"].between(
        0, 180
    ).all():
        raise ValueError(
            "impact_height must be between 0 and 180."
        )

    if not df["ball_length"].between(
        0.5, 11
    ).all():
        raise ValueError(
            "ball_length must be between 0.5 and 11."
        )

    if not df["ball_line"].between(
        -120, 160
    ).all():
        raise ValueError(
            "ball_line must be between -120 and 160."
        )

    if not df["timing"].between(
        0, 1
    ).all():
        raise ValueError(
            "timing must be between 0 and 1."
        )

    if not df["front_foot"].isin([0, 1]).all():
        raise ValueError(
            "front_foot must be either 0 or 1."
        )