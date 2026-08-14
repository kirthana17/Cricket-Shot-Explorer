from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = ROOT / "data" / "shots.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_ROWS = 5000


# ============================================================
# SHOT CLASSES
# ============================================================

SHOTS = [
    "cover_drive",
    "straight_drive",
    "flick",
    "pull",
    "cut",
    "defensive",
]


# ============================================================
# FEATURE PROFILES
# ============================================================

PROFILES = {

    "cover_drive": {
        "bat_speed": (95, 10),
        "impact_height": (35, 12),
        "ball_length": (3.2, 0.5),
        "ball_line": (55, 20),
        "timing": (0.75, 0.12),
        "front_foot": 0.90,
    },

    "straight_drive": {
        "bat_speed": (93, 11),
        "impact_height": (38, 12),
        "ball_length": (3.0, 0.5),
        "ball_line": (15, 18),
        "timing": (0.76, 0.12),
        "front_foot": 0.92,
    },

    "flick": {
        "bat_speed": (88, 12),
        "impact_height": (48, 14),
        "ball_length": (3.8, 0.6),
        "ball_line": (-45, 22),
        "timing": (0.72, 0.14),
        "front_foot": 0.65,
    },

    "pull": {
        "bat_speed": (98, 12),
        "impact_height": (72, 16),
        "ball_length": (6.5, 0.7),
        "ball_line": (-10, 25),
        "timing": (0.70, 0.15),
        "front_foot": 0.20,
    },

    "cut": {
        "bat_speed": (91, 11),
        "impact_height": (68, 15),
        "ball_length": (6.0, 0.7),
        "ball_line": (65, 25),
        "timing": (0.73, 0.14),
        "front_foot": 0.25,
    },

    "defensive": {
        "bat_speed": (75, 15),
        "impact_height": (40, 15),
        "ball_length": (4.2, 0.9),
        "ball_line": (10, 30),
        "timing": (0.55, 0.18),
        "front_foot": 0.70,
    },
}


# ============================================================
# DATA GENERATION FUNCTION
# ============================================================

def generate(
    n_rows=N_ROWS,
    random_state=RANDOM_STATE,
):

    rng = np.random.default_rng(random_state)

    rows_per_class = n_rows // len(SHOTS)

    remainder = n_rows % len(SHOTS)

    rows = []

    for index, shot in enumerate(SHOTS):

        count = rows_per_class + (
            1 if index < remainder else 0
        )

        profile = PROFILES[shot]

        class_df = pd.DataFrame({

            "bat_speed": rng.normal(
                profile["bat_speed"][0],
                profile["bat_speed"][1],
                count,
            ),

            "impact_height": rng.normal(
                profile["impact_height"][0],
                profile["impact_height"][1],
                count,
            ),

            "ball_length": rng.normal(
                profile["ball_length"][0],
                profile["ball_length"][1],
                count,
            ),

            "ball_line": rng.normal(
                profile["ball_line"][0],
                profile["ball_line"][1],
                count,
            ),

            "timing": rng.normal(
                profile["timing"][0],
                profile["timing"][1],
                count,
            ),

            "front_foot": (
                rng.random(count)
                < profile["front_foot"]
            ).astype(int),

            "shot_type": shot,
        })

        rows.append(class_df)

    df = pd.concat(
        rows,
        ignore_index=True,
    )

    # Keep generated values within sensible ranges.
    df["bat_speed"] = (
        df["bat_speed"]
        .clip(20, 150)
        .round(2)
    )

    df["impact_height"] = (
        df["impact_height"]
        .clip(0, 180)
        .round(2)
    )

    df["ball_length"] = (
        df["ball_length"]
        .clip(0.5, 11)
        .round(3)
    )

    df["ball_line"] = (
        df["ball_line"]
        .clip(-120, 160)
        .round(2)
    )

    df["timing"] = (
        df["timing"]
        .clip(0, 1)
        .round(4)
    )

    # Shuffle the rows so the classes are not grouped together.
    df = df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    return df


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    df = generate()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Generated {len(df)} rows."
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("\nClass distribution:")

    print(
        df["shot_type"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()