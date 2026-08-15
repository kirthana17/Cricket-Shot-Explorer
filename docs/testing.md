# Testing Strategy

## Unit Tests

The project includes automated tests for the core dataset and model artifacts.

The test suite verifies:

- Dataset existence
- Dataset shape
- Dataset columns
- Presence of all six shot classes
- Missing values
- Feature ranges
- Model artifact existence
- Model prediction functionality

The automated tests are implemented in:

`tests/test_project.py`

---

## Dataset Testing

The dataset tests verify that:

- The dataset file exists.
- The dataset contains exactly 5,000 observations.
- All required feature columns are present.
- The target column `shot_type` is present.
- All six expected shot classes are represented.
- No missing values are present.
- Numerical features remain within their defined ranges.
- `front_foot` contains only valid binary values (`0` or `1`).

---

## Model Testing

The model is tested by verifying that:

- The trained model artifact exists.
- The saved model can be loaded successfully.
- A valid sample input can be passed to the model.
- The model produces a prediction.
- The prediction belongs to one of the six expected shot classes.

---

## Application Testing

The Streamlit dashboard is manually tested by:

1. Launching the application.
2. Checking that the application loads successfully.
3. Testing the Prediction page.
4. Moving each feature control.
5. Confirming that predictions are generated.
6. Checking the EDA page.
7. Checking the Model Performance page.
8. Checking the confusion matrix.
9. Checking model metrics and visualizations.

---

## Integration Testing

The complete machine-learning workflow is verified through the project
pipeline:

```text
Dataset
   |
   v
Feature Preparation
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Saved Model
   |
   v
Streamlit Prediction

This confirms that the individual components work together as an
end-to-end system.

Test Results

The automated test suite currently passes all tests:

8 passed

The passing tests confirm that the dataset structure, data quality, model
artifact, and prediction functionality satisfy the defined requirements.

Reproducibility

Reproducibility is supported through the use of a fixed random seed:

42

The seed is used during dataset generation, train-test splitting, and model
training where applicable.

This allows the dataset-generation and model-development process to be
reproduced consistently.

Acceptance Criteria

The project is considered functional when:

Dataset generation succeeds.
Dataset validation succeeds.
Model training succeeds.
Model evaluation succeeds.
Automated tests pass.
The trained model artifact is created.
The Streamlit application launches successfully.
User inputs produce predictions.
EDA visualizations are available.
Model performance information is displayed.
Documentation allows another user to reproduce the project.