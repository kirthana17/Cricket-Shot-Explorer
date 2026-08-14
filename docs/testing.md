# Testing Strategy

## Unit Tests

The project tests:

- Dataset shape
- Dataset columns
- Class presence
- Missing values
- Feature ranges
- Reproducibility

## Model Testing

The training pipeline is verified by checking that:

- Training completes successfully.
- Evaluation metrics are produced.
- The model artifact is created.
- Predictions can be generated.

## Application Testing

The Streamlit dashboard is manually tested by:

1. Launching the application.
2. Moving each feature control.
3. Confirming the prediction changes.
4. Checking all three tabs.
5. Checking the confusion matrix.
6. Checking model metrics.

## Acceptance Criteria

The project is considered functional when:

- Dataset generation succeeds.
- Training succeeds.
- Evaluation succeeds.
- Tests pass.
- Streamlit launches.
- User inputs produce predictions.
- Documentation allows another user to reproduce the project.