# Cricket Shot Explorer

An end-to-end machine learning system for cricket shot classification.

## Overview

Cricket Shot Explorer predicts the type of cricket shot played by a batter using six structured features:

- Bat speed
- Impact height
- Ball length
- Ball line
- Timing
- Front-foot usage

The system performs the complete machine learning workflow:

Data Generation → Feature Preparation → Model Training → Model Evaluation → Model Selection → Prediction → Visualization → Deployment

## Shot Classes

The system predicts six shot types:

1. Cover Drive
2. Straight Drive
3. Flick
4. Pull
5. Cut
6. Defensive

## Machine Learning Problem

This is a supervised multi-class classification problem.

### Input

Six features describing the delivery and bat-ball interaction.

### Output

One of six cricket shot classes.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Pytest

## Project Structure

```text
Cricket-Shot-Explorer/
│
├── app/
│   ├── assets/
│   ├── components/
│   └── pages/
│
├── configs/
│
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
│
├── docs/
│   ├── data_dictionary.md
│   ├── ml_design.md
│   ├── requirements.md
│   ├── system_design.md
│   └── testing.md
│
├── models/
├── notebooks/
├── reports/
│
├── src/
│   ├── data
│   ├── evaluation
│   ├── features
│   ├── models
│   ├── utils
│   └── visualization
│
├── tests/
│
├── app.py
├── LICENSE
├── README.md
└── requirements.txt