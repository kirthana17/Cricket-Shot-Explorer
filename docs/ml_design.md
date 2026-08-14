# Machine Learning Design

## Problem Type

Multi-class classification.

The target variable is `shot_type`, which contains six possible classes.

## Features

1. bat_speed
2. impact_height
3. ball_length
4. ball_line
5. timing
6. front_foot

## Target

`shot_type`

## Dataset

The dataset is synthetic and contains approximately 5,000 observations.

A fixed random seed is used so that the generated dataset is reproducible.

## Models

### Logistic Regression

Used as the simple baseline.

### Random Forest

Used to determine whether a non-linear model provides a meaningful improvement.

## Data Split

80% training data.

20% held-out testing data.

Stratification is used to preserve class proportions.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- 5-fold cross-validation accuracy

## Model Selection

If the best model improves over Logistic Regression by less than 1 percentage
point, Logistic Regression is preferred because it is simpler, faster and easier
to interpret.

## Current Result

Both Logistic Regression and Random Forest achieve approximately 83.5%
accuracy on the held-out test set.

Therefore Logistic Regression is selected.

## Limitations

The dataset is synthetic and does not represent real cricket data.

The six features do not contain all information needed to distinguish similar
shots.