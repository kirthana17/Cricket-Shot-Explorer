# Machine Learning Design

## Problem Type

Multi-class classification.

The target variable is `shot_type`, which contains six possible classes:

- `cover_drive`
- `straight_drive`
- `flick`
- `pull`
- `cut`
- `defensive`

---

## Objective

The objective is to build a machine-learning system that predicts the type of
cricket shot from a set of structured input features representing the delivery
and shot characteristics.

The system is designed as an end-to-end machine-learning pipeline covering
data generation, feature preparation, model training, model comparison,
evaluation, and prediction.

---

## Features

The model uses six input features:

1. `bat_speed`
2. `impact_height`
3. `ball_length`
4. `ball_line`
5. `timing`
6. `front_foot`

### Feature Types

| Feature | Type |
|---|---|
| `bat_speed` | Numerical |
| `impact_height` | Numerical |
| `ball_length` | Numerical |
| `ball_line` | Numerical |
| `timing` | Numerical |
| `front_foot` | Binary |

---

## Target

The target variable is:

`shot_type`

It contains six classes:

- `cover_drive`
- `straight_drive`
- `flick`
- `pull`
- `cut`
- `defensive`

---

## Dataset

The dataset is synthetic and contains exactly 5,000 observations.

The dataset is generated using predefined statistical profiles for each shot
class.

Numerical features are sampled from normal distributions, while
`front_foot` is generated using class-specific probabilities.

A fixed random seed of `42` is used to make the dataset reproducible.

The generated dataset is stored at:

`data/shots.csv`

---

## Data Preparation

The feature preparation pipeline:

1. Loads the generated dataset.
2. Validates that all required features are present.
3. Selects the six model features.
4. Converts `front_foot` to an integer binary representation.
5. Separates the input features from the target variable.

The feature preparation logic is implemented in:

`src/features.py`

---

## Data Split

The dataset is divided using an 80/20 stratified train-test split.

```text
Training set: 4,000 observations
Test set:     1,000 observations

Stratification is used to preserve the class distribution between the training
and held-out test sets.

A fixed random state of 42 is used.

Models

Two classification models are trained and compared.

Logistic Regression

Logistic Regression is used as the primary baseline model.

A StandardScaler is applied before Logistic Regression using a scikit-learn
pipeline.

The model provides:

A simple baseline
Fast training
Relatively easy interpretation
A reference point for model comparison
Random Forest

Random Forest is used as the non-linear comparison model.

The implementation uses:

300 decision trees
Minimum samples per leaf of 2
Random state of 42
Parallel training using all available CPU cores

The purpose of Random Forest is to determine whether a non-linear ensemble
model provides a meaningful improvement over Logistic Regression.

Cross-Validation

Five-fold cross-validation is performed on the training dataset.

Cross-validation is used to estimate how consistently each model performs
across different subsets of the training data.

The primary cross-validation metric is accuracy.

Evaluation Metrics

The following metrics are calculated:

Accuracy
Precision
Recall
F1 score
Macro F1
Confusion matrix
Five-fold cross-validation accuracy

The confusion matrix is used to identify which shot classes are most commonly
confused with one another.

Model Selection

Logistic Regression is preferred unless Random Forest provides a meaningful
improvement.

The selection rule is:

Random Forest must improve test accuracy by more than 1 percentage point
over Logistic Regression to replace the Logistic Regression baseline.

Otherwise, Logistic Regression is selected because it is simpler, faster,
and easier to interpret.

Model Results

The current evaluation produced the following results:

Model	Test Accuracy	CV Mean Accuracy	Macro F1
Logistic Regression	86.00%	86.50%	0.859
Random Forest	85.60%	85.80%	0.855

Since Random Forest does not improve over Logistic Regression by more than
1 percentage point, Logistic Regression is selected as the final model.

The selected model is saved as:

models/model.joblib

Final Evaluation

The saved model is evaluated on the held-out 20% test split.

This corresponds to:

1,000 held-out test observations

The final selected model achieved:

Metric	Result
Test Accuracy	86.00%
Macro F1	0.8593

The evaluation results are saved to:

reports/evaluation_results.json

Feature Importance

Random Forest feature importance is calculated during model comparison to
provide additional insight into which features contribute to the non-linear
model.

The feature-importance information is stored in:

reports/metrics.json

and displayed in the Streamlit Model Performance page.

Confusion Matrix

A confusion matrix is generated for the selected model.

It shows the relationship between:

Actual shot classes
Predicted shot classes

The visualization is saved as:

reports/confusion_matrix.png

The confusion matrix is also displayed in the Streamlit application.

Reproducibility

The pipeline uses a fixed random seed:

42

The seed is used for dataset generation, train-test splitting, and model
training where applicable.

This allows the dataset and model-development process to be reproduced
consistently.

Limitations

The dataset is synthetic and does not represent real cricket data.

The model currently uses only six structured features and does not process:

Video
Images
Player pose
Ball-tracking data
Bat trajectory
Player-specific characteristics
Match context
Bowling characteristics

The generated feature distributions are predefined and do not capture the
full complexity of real cricket.

Therefore, the reported accuracy should only be interpreted as performance
on the synthetic dataset and not as real-world cricket-shot prediction
accuracy.

Future Improvements

Potential improvements include:

Collecting real cricket shot data.
Incorporating video-based features.
Using pose estimation to extract player movement.
Adding ball trajectory and bowling characteristics.
Incorporating player-specific information.
Testing gradient boosting and other advanced models.
Performing hyperparameter optimization.
Adding model explainability techniques.
Evaluating the system on an independent real-world dataset.