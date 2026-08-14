# 🏏 Cricket Shot Explorer

An end-to-end Machine Learning application that predicts cricket shot type from structured batting and ball-delivery features.

The project covers the complete ML workflow — from data generation and exploratory data analysis to feature engineering, model training, evaluation, interpretability, testing, and deployment as an interactive Streamlit application.

> **Note:** The current version uses a synthetic structured dataset for demonstration and system development. The predictions should therefore not be interpreted as real-world cricket biomechanics.

---

## 🚀 Live Demo

**Live Application:**  
https://cricket-shot-explorer.streamlit.app/

The deployed application allows users to:

- Enter cricket shot-related parameters
- Generate a shot-type prediction
- View prediction confidence
- Inspect class probabilities
- Explore the dataset
- Analyze feature distributions
- View feature correlations
- Compare ML models
- Inspect the confusion matrix
- Explore feature importance

---

# 📌 Project Overview

Cricket shot selection depends on multiple interacting factors such as bat speed, impact position, ball length, ball line, timing, and foot movement.

This project explores whether these structured features can be used to classify a cricket shot into one of six predefined shot categories.

The system implements a complete machine-learning pipeline:

```text
Input Features
      ↓
Data Validation
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Cross-Validation
      ↓
Model Evaluation
      ↓
Model Selection
      ↓
Saved ML Pipeline
      ↓
Streamlit Application
      ↓
Prediction + Explainability

🎯 Problem Statement

Develop a machine-learning system capable of classifying a cricket shot using structured input features representing batting and ball-delivery conditions.

The system should:

Accept structured cricket-related inputs.
Process the inputs consistently with the training pipeline.
Predict the most likely shot type.
Provide prediction probabilities.
Evaluate multiple machine-learning models.
Provide visual analysis of the dataset.
Provide interpretable model information.
Be deployable as a web application.
🎯 Objectives

The main objectives of the project are:

Build an end-to-end supervised ML classification pipeline.
Perform exploratory data analysis.
Analyze relationships between cricket-related features.
Train and compare multiple classification models.
Evaluate model performance using appropriate metrics.
Save the trained model for inference.
Build an interactive Streamlit interface.
Provide prediction confidence and class probabilities.
Provide model interpretability through feature importance.
Implement automated project tests.
Deploy the application publicly.
📊 Dataset

The current implementation uses a synthetic structured dataset containing 5,000 observations.

The dataset contains six input features and one target variable.

Dataset characteristics
Property	Value
Number of rows	5,000
Input features	6
Target classes	6
Dataset type	Synthetic structured data
Task	Multi-class classification
🏏 Shot Classes

The target variable contains six shot categories:

cover_drive
cut
defensive
flick
pull
straight_drive
🔢 Features

The model uses the following six input features:

Feature	Description	Unit / Type
bat_speed	Speed of the bat during the shot	km/h
impact_height	Height at which bat-ball impact occurs	cm
ball_length	Length of the delivered ball	m
ball_line	Position/line of the delivery	cm
timing	Timing-related shot parameter	numerical
front_foot	Whether the shot uses the front foot	binary
🤖 Machine Learning Models

The project currently evaluates multiple classification models.

Logistic Regression

Logistic Regression is used as one of the primary classification models.

It provides:

Multi-class classification
Probability estimates
A relatively interpretable baseline
Efficient training and inference
Random Forest

Random Forest is also evaluated as a non-linear tree-based model.

It provides:

Non-linear decision boundaries
Feature importance
Robust classification performance
An additional model for comparison
📈 Model Performance

The current evaluation produced the following results:

Model	Test Accuracy	CV Accuracy	CV Std	Macro F1
Logistic Regression	86.0%	86.5%	0.92%	0.859
Random Forest	85.6%	85.8%	0.82%	0.855

Based on the current evaluation, Logistic Regression is selected as the final prediction model.

The model achieved:

Test Accuracy: 86.0%
Cross-Validation Accuracy: 86.5%
Macro F1: 0.859

Because the dataset is synthetic, these metrics should be interpreted as evaluation results for this experimental dataset rather than evidence of real-world cricket prediction accuracy.

🔬 Exploratory Data Analysis

The application includes an interactive EDA section.

Available analyses
Shot class distribution
Feature distributions
Feature distributions by shot type
Correlation matrix
Numerical feature exploration

Example feature analyses include:

Bat Speed by Shot Type
Impact Height by Shot Type
Ball Length by Shot Type
Ball Line by Shot Type
Timing by Shot Type
Front Foot by Shot Type
🔗 Feature Correlation

The project calculates correlations between the six numerical/binary input features.

An example of the observed relationships is:

impact_height and ball_length: approximately 0.61
ball_length and front_foot: approximately -0.52
impact_height and front_foot: approximately -0.41

The correlation matrix is provided in the EDA section of the application.

📊 Model Evaluation

The model evaluation section provides:

Accuracy

Measures the proportion of correctly classified samples.

Cross-Validation Accuracy

Evaluates model performance across multiple validation folds.

Cross-Validation Standard Deviation

Shows variation in cross-validation performance.

Macro F1 Score

Calculates the F1 score independently for each class and averages the results, giving equal importance to each shot category.

🧩 Confusion Matrix

The application provides a multi-class confusion matrix to examine how predictions are distributed across the six shot classes.

The matrix helps identify:

Correct classifications
Frequently confused shot types
Class-specific prediction behaviour
Areas where the classifier could be improved
🔍 Feature Importance

Feature importance is calculated using the Random Forest model for interpretability.

The current application visualizes the relative importance of:

ball_line
ball_length
bat_speed
impact_height
front_foot
timing

This interpretability analysis is displayed even when Logistic Regression is selected as the final prediction model.

🖥️ Application Features

The Streamlit application contains three major sections.

1. 🎯 Prediction

Users can provide:

Bat speed
Impact height
Ball length
Ball line
Timing
Front-foot selection

The application then displays:

Predicted Shot
Prediction Confidence
Prediction Probabilities
2. 📊 EDA

The EDA section provides:

Dataset statistics
Shot class distribution
Feature distributions
Interactive feature selection
Feature correlation matrix
3. 📈 Model Performance

The model performance section provides:

Selected model
Test accuracy
Cross-validation accuracy
Model comparison
Confusion matrix
Feature importance
🏗️ System Architecture
                    ┌──────────────────────┐
                    │     User Input       │
                    │  Cricket Parameters  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Input Validation   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Processing   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Saved ML Pipeline    │
                    │ Logistic Regression  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Prediction       │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌──────────────────┐
        │ Predicted Shot  │        │ Class Probabilities│
        └─────────────────┘        └──────────────────┘
📁 Project Structure
Cricket-Shot-Explorer/
│
├── app.py
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── app/
│   ├── assets/
│   ├── components/
│   └── pages/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── external/
│   ├── processed/
│   ├── raw/
│   └── shots.csv
│
├── docs/
│   ├── data_dictionary.md
│   ├── ml_design.md
│   ├── requirements.md
│   ├── system_design.md
│   └── testing.md
│
├── models/
│   └── model.joblib
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   ├── evaluation_results.json
│   └── metrics.json
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── visualization/
│
└── tests/
    └── test_project.py
⚙️ Technology Stack
Programming Language
Python
Machine Learning
Scikit-learn
Joblib
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Application
Streamlit
Testing
Pytest
Development
VS Code
Git
GitHub
Deployment
Streamlit Community Cloud
🛠️ Installation
1. Clone the repository
git clone https://github.com/kirthana17/Cricket-Shot-Explorer.git

Move into the project directory:

cd Cricket-Shot-Explorer
2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
▶️ Running the Application Locally

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

🧪 Running Tests

Run the automated test suite using:

pytest

For more detailed output:

pytest -v
📓 Running the EDA Notebook

The exploratory analysis notebook is located at:

notebooks/01_eda.ipynb

The notebook contains the exploratory analysis used to understand the dataset and its feature relationships.

📦 Model Artifact

The trained model is stored at:

models/model.joblib

The application loads this saved model during inference rather than retraining the model every time a user accesses the application.

🔄 End-to-End ML Workflow

The project follows this workflow:

1. Dataset Generation
        ↓
2. Data Validation
        ↓
3. Exploratory Data Analysis
        ↓
4. Feature Analysis
        ↓
5. Train/Test Split
        ↓
6. Model Training
        ↓
7. Cross Validation
        ↓
8. Model Comparison
        ↓
9. Model Selection
        ↓
10. Model Serialization
        ↓
11. Application Integration
        ↓
12. Automated Testing
        ↓
13. Cloud Deployment
🧪 Testing Strategy

The project contains automated tests under:

tests/

Testing focuses on validating important parts of the ML application, including:

Project structure
Data availability
Model availability
Prediction functionality
Expected project outputs

Run:

pytest -v
🌐 Deployment

The application is deployed using Streamlit Community Cloud.

The deployment uses the GitHub repository as the source code and installs dependencies from:

requirements.txt

The application entry point is:

app.py
⚠️ Current Limitations

The current system has several limitations.

1. Synthetic Dataset

The dataset is synthetic rather than collected from real cricket matches or biomechanical sensors.

Therefore:

Model performance cannot be interpreted as real-world cricket-shot prediction accuracy.

2. Structured Inputs

The application currently predicts shots from manually entered numerical features.

It does not directly process:

Video
Images
Pose estimation
Ball-tracking data
Bat-tracking data
Player tracking data
3. Limited Feature Set

Only six structured features are currently used.

Real-world cricket shot classification would likely require additional contextual and biomechanical information.

🚀 Future Improvements

Potential future improvements include:

Real Cricket Dataset

Replace synthetic data with:

Annotated cricket videos
Ball-tracking data
Player pose data
Bat trajectory data
Match datasets
Computer Vision

Introduce video-based shot classification using:

OpenCV
Pose estimation
Object detection
Temporal deep learning
Deep Learning

Potential models include:

CNN
LSTM
CNN-LSTM
Transformer-based video models
Real-Time Prediction

Develop a system capable of processing live cricket video and predicting the shot automatically.

Advanced Explainability

Add:

SHAP
LIME
Per-prediction explanations
Model Monitoring

Add production monitoring for:

Prediction distributions
Input drift
Data drift
Model performance
Failure cases
📚 Documentation

Additional technical documentation is available in:

docs/
Data Dictionary
docs/data_dictionary.md

Contains descriptions of dataset fields.

ML Design
docs/ml_design.md

Describes the machine-learning design and modelling approach.

System Design
docs/system_design.md

Describes the overall application architecture and prediction flow.

Requirements
docs/requirements.md

Documents project requirements.

Testing
docs/testing.md

Documents the testing strategy.

🔐 Reproducibility

The project keeps the machine-learning workflow organized into separate components for:

Data processing
Feature processing
Model training
Evaluation
Visualization
Application serving

This separation makes the system easier to reproduce, test, maintain, and extend.

📌 Project Status
Current Status: ✅ Deployed

Implemented:

 Dataset
 Data processing
 Exploratory data analysis
 Feature analysis
 Logistic Regression
 Random Forest
 Cross-validation
 Model comparison
 Confusion matrix
 Feature importance
 Model serialization
 Streamlit application
 Automated tests
 Git version control
 GitHub repository
 Cloud deployment

Planned improvements:

 Real-world cricket dataset
 Video-based shot classification
 Computer vision pipeline
 Advanced explainability
 Model monitoring
 Real-time prediction

👩‍💻 Author

Kirthana S

GitHub:

https://github.com/kirthana17

📄 License

This project is licensed under the terms specified in the LICENSE file.

⭐ Acknowledgement

This project was developed as an end-to-end machine-learning engineering project, combining data processing, machine learning, model evaluation, visualization, software testing, and cloud deployment into a single application.