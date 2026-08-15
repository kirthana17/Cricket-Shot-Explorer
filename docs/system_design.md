# System Design

## High-Level Architecture

```text
Synthetic Dataset
       |
       v
Data Generation
       |
       v
Data Validation
       |
       v
Exploratory Data Analysis
       |
       v
Feature Preparation
       |
       v
Model Training
       |
       +------------------+
       |                  |
       v                  v
Logistic Regression   Random Forest
       |                  |
       +--------+---------+
                |
                v
        Model Evaluation
                |
                v
          Model Selection
                |
                v
          Saved Model
         model.joblib
                |
                v
       Prediction Pipeline
                |
                v
      Streamlit Dashboard
                |
                v
             End User

System Components
1. Data Generation

The synthetic dataset is generated using predefined statistical profiles for
the six cricket shot classes.

The generator creates 5,000 observations using six input features and one
target variable.

Implementation:

src/data.py

Output:

data/shots.csv

2. Data Validation and Feature Preparation

The dataset is validated to ensure that the required features are present and
that the input values follow the expected structure.

The feature preparation process selects the six model features and converts
front_foot into an integer binary representation.

Implementation:

src/features.py

3. Exploratory Data Analysis

The dataset is explored through class-distribution plots, feature
distributions, and a correlation matrix.

The generated visualizations are stored in the reports/ directory and are
also used by the Streamlit dashboard.

Implementation:

src/visualize.py

4. Model Training

Two classification models are trained:

Logistic Regression
Random Forest

Logistic Regression uses feature scaling through a StandardScaler pipeline.

Random Forest provides a non-linear comparison model.

Implementation:

src/models.py

5. Model Evaluation

The trained models are evaluated using:

Test accuracy
Precision
Recall
F1 score
Macro F1
Confusion matrix
Five-fold cross-validation accuracy

The evaluation results are stored as JSON reports.

Implementation:

src/evaluation.py

6. Model Selection

The models are compared based primarily on held-out test accuracy.

Random Forest replaces Logistic Regression only when its accuracy improves by
more than one percentage point.

Otherwise, Logistic Regression is selected because of its simplicity and
interpretability.

The selected model is stored as:

models/model.joblib

7. Prediction Pipeline

The prediction pipeline accepts six feature values:

bat_speed
impact_height
ball_length
ball_line
timing
front_foot

The values are converted into a pandas DataFrame and passed to the saved
machine-learning pipeline.

The pipeline returns the predicted shot class.

Training Flow
shots.csv
   |
   v
Load Dataset
   |
   v
Feature / Label Separation
   |
   v
80/20 Stratified Split
   |
   v
Model Training
   |
   +----------------------+
   |                      |
   v                      v
Logistic Regression    Random Forest
   |                      |
   +----------+-----------+
              |
              v
       Five-Fold CV
              |
              v
       Model Evaluation
              |
              v
       Model Selection
              |
              v
         model.joblib

Prediction Flow
User
 |
 v
Six Input Features
 |
 v
Input Validation
 |
 v
Prediction DataFrame
 |
 v
Saved Model Pipeline
 |
 v
Prediction
 |
 v
Predicted Shot Type
Application Flow

The Streamlit application provides three main functional areas:

Streamlit Application
        |
        +-------------------+
        |         |         |
        v         v         v
   Prediction     EDA    Performance
        |         |         |
        v         v         v
    Predict     Explore   Evaluate
    Shot Type   Dataset    Model
Prediction Page

Allows the user to enter the six model features and obtain a predicted
cricket shot type.

EDA Page

Displays the dataset structure and generated exploratory visualizations.

Model Performance Page

Displays model evaluation information, including accuracy, classification
metrics, confusion matrix, and feature-importance information.

Project Artifacts
Artifact	Purpose
data/shots.csv	Generated dataset
models/model.joblib	Selected trained model
reports/metrics.json	Model comparison metrics
reports/evaluation_results.json	Evaluation results
reports/confusion_matrix.png	Confusion matrix visualization
reports/*.png	EDA visualizations
Testing Flow

The automated test suite validates important project components:

Project
   |
   +--> Dataset Validation
   |
   +--> Class Validation
   |
   +--> Feature Range Validation
   |
   +--> Model Artifact Validation
   |
   +--> Model Prediction Validation

Tests are implemented in:

tests/test_project.py

Deployment Architecture

The trained model is stored as a reusable joblib artifact.

The Streamlit application loads the saved model and uses it for inference.

User Browser
      |
      v
Streamlit Application
      |
      v
Saved ML Pipeline
      |
      v
Prediction
      |
      v
Shot Classification Result

The application can therefore perform predictions without retraining the
model during normal inference.