# Data Dictionary — Cricket Shot Explorer 
 
This document describes the synthetic structured dataset used by the Cricket Shot Explorer machine-learning pipeline. 
 
The dataset is generated programmatically by `src/data.py` and contains 5,000 observations representing six predefined cricket shot classes. 
 
> **Note:** The dataset is synthetic and is intended for machine-learning pipeline development and demonstration. It does not represent real-world cricket biomechanics or match data. 
 
--- 
 
## Dataset Overview 
 
| Property | Value | 
|---|---| 
| Number of observations | 5,000 | 
| Input features | 6 | 
| Target variable | `shot_type` | 
| Number of classes | 6 | 
| Dataset type | Synthetic structured data | 
| Task | Multi-class classification | 
| Random seed | 42 | 
 
--- 
 
## Columns 
 
| Column | Description | Unit / Type | Example | 
|---|---|---|---| 
| `bat_speed` | Speed of the bat during the shot | km/h | `95` | 
| `impact_height` | Height of bat-ball impact | cm | `35` | 
| `ball_length` | Length of the delivered ball | m | `3.2` | 
| `ball_line` | Horizontal position of the delivery relative to the stumps | cm | `55` | 
| `timing` | Timing quality of the shot | 0–1 | `0.75` | 
| `front_foot` | Whether the batter uses the front foot | Binary (0/1) | `1` | 
| `shot_type` | Target class representing the shot played | Categorical | `cover_drive` | 
 
--- 
 
## Feature Interpretation 
 
### `bat_speed` 
 
Represents the estimated speed of the bat during the shot. 
 
Higher values generally represent more aggressive shots within the synthetic dataset. 
 
### `impact_height` 
 
Represents the approximate height at which bat-ball contact occurs. 
 
### `ball_length` 
 
Represents the approximate length of the delivery. 
 
Smaller values represent fuller deliveries, while larger values represent shorter deliveries. 
 
### `ball_line` 
 
Represents the horizontal position of the delivery relative to the stumps. 
 
- `0` = approximately around the stumps 
- Positive values = off side 
- Negative values = leg side 
 
### `timing` 
 
Represents the quality of shot timing on a normalized scale. 
 
`0 ≤ timing ≤ 1` 
 
### `front_foot` 
 
Binary feature indicating front-foot usage. 
 
- `0` = not front foot 
- `1` = front foot 
 
--- 
 
## Shot Classes 
 
The target variable `shot_type` contains six classes: 
 
- `cover_drive` 
- `straight_drive` 
- `flick` 
- `pull` 
- `cut` 
- `defensive` 
 
--- 
 
## Synthetic Shot Profiles 
 
The following profiles are implemented in `src/data.py`. 
 
Values for numerical features are represented as mean (standard deviation). 
 
`front_foot` represents the probability of front-foot usage. 
 
| Shot Type | Bat Speed | Impact Height | Ball Length | Ball Line | Timing | Front Foot | 
|---|---:|---:|---:|---:|---:|---:| 
| `cover_drive` | 95 (10) | 35 (12) | 3.2 (0.5) | +55 (20) | 0.75 (0.12) | 90% | 
| `straight_drive` | 93 (11) | 38 (12) | 3.0 (0.5) | +15 (18) | 0.76 (0.12) | 92% | 
| `flick` | 88 (12) | 48 (14) | 3.8 (0.6) | -45 (22) | 0.72 (0.14) | 65% | 
| `pull` | 98 (12) | 72 (16) | 6.5 (0.7) | -10 (25) | 0.70 (0.15) | 20% | 
| `cut` | 91 (11) | 68 (15) | 6.0 (0.7) | +65 (25) | 0.73 (0.14) | 25% | 
| `defensive` | 75 (15) | 40 (15) | 4.2 (0.9) | +10 (30) | 0.55 (0.18) | 70% | 
 
--- 
 
## Feature Bounds 
 
Generated values are clipped to the following ranges: 
 
| Feature | Minimum | Maximum | 
|---|---:|---:| 
| `bat_speed` | 20 | 150 | 
| `impact_height` | 0 | 180 | 
| `ball_length` | 0.5 | 11 | 
| `ball_line` | -120 | 160 | 
| `timing` | 0 | 1 | 
| `front_foot` | 0 | 1 | 
 
--- 
 
## Data Generation 
 
The dataset is generated using: 
 
1. Class-specific feature profiles. 
2. Normal distributions for numerical features. 
3. Bernoulli sampling for `front_foot`. 
4. Feature-range clipping. 
5. Numerical rounding. 
6. Class labeling. 
7. Random shuffling. 
 
The generator uses: 
 
```python
RANDOM_STATE = 42
N_ROWS = 5000

The generated dataset is saved to:

`data/shots.csv`

---

## Class Distribution

The dataset is approximately balanced across the six classes.

With 5,000 observations distributed across six classes, each class contains approximately 833 or 834 observations.

This provides a balanced classification dataset for model development and comparison.

---

## Intentional Class Overlap

The synthetic dataset intentionally contains overlapping feature distributions between similar shot types.

Important overlaps include:

1. `cover_drive` and `straight_drive`
2. `flick` and `defensive`
3. `pull` and `cut`

This prevents the synthetic classification problem from becoming unrealistically easy and allows meaningful model comparison and evaluation.

---

## Data Quality

The generated dataset is validated for:

- Required feature columns
- Correct number of observations
- Presence of all six shot classes
- Missing values
- Valid feature ranges
- Valid binary `front_foot` values

These properties are also verified by the automated tests in:

`tests/test_project.py`

---

## Dataset Limitations

This dataset is synthetic and should not be interpreted as real cricket biomechanical data.

The current generator does not model:

- Player-specific characteristics
- Bat trajectory
- Ball trajectory
- Player pose
- Match context
- Bowling speed
- Spin
- Real sensor measurements
- Video information

The numerical features are generated from predefined statistical profiles. Therefore, the dataset is primarily intended to demonstrate an end-to-end machine-learning workflow.

---

## Model Evaluation Reference

The machine-learning pipeline uses an 80/20 stratified train-test split.

- Training set: 4,000 observations
- Test set: 1,000 observations

The selected Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | 86.00% |
| Macro F1 | 0.8593 |

Five-fold cross-validation during model development achieved approximately 86.5% mean accuracy for Logistic Regression.

These results describe performance on the synthetic dataset and should not be interpreted as real-world cricket-shot prediction accuracy.

---

## Source Files

Dataset generation:

`src/data.py`

Dataset:

`data/shots.csv`

Feature processing:

`src/features.py`

Model training:

`src/models.py`

Evaluation:

`src/evaluation.py`

Testing:

`tests/test_project.py`