# Software Requirements

## Problem Statement

Manual cricket shot classification can be time-consuming and subjective.
Different analysts may classify similar shots differently depending on their
experience.

## Objective

Build an end-to-end machine-learning system that predicts one of six cricket
shot types from six measurable input features.

## Functional Requirements

1. Generate a synthetic dataset.
2. Explore and visualize the dataset.
3. Train multiple classification models.
4. Evaluate model performance.
5. Select and save the best model.
6. Accept six feature inputs from a user.
7. Predict the cricket shot type.
8. Display model performance.
9. Provide an interactive Streamlit dashboard.

## Non-Functional Requirements

1. Reproducibility.
2. Reasonable prediction speed.
3. Usable interface.
4. Maintainable code structure.
5. Clear documentation.
6. Testable components.

## In Scope

- Synthetic data generation
- EDA
- Classification
- Evaluation
- Streamlit dashboard
- Testing
- Deployment

## Out of Scope

- Real cricket video processing
- Real-time ball tracking
- Pose estimation from video
- Real player biometric data
- Production-grade sports analytics

## Constraints

- Six predefined features.
- Six predefined shot classes.
- Synthetic data only.
- Approximately 5,000 observations.
- Limited project timeline.

## Stakeholders

- Cricket coaches
- Players
- Sports analysts
- Developers
- Project mentor