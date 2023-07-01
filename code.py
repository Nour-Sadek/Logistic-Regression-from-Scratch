# Imported packages
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""Stage 1: Sigmoid function

Description

In this project, we will work on a classification algorithm that makes 
predictions when a dependent variable assumes discrete values. Logistic 
regression is arguably the simplest solution. In the case of binary 
classification (class 0 or class 1), it uses a sigmoid function to estimate how 
likely an observation belongs to class 1.

we will work with the Wisconsin Breast Cancer Dataset from the sklearn library.
We also want to standardize the features as they are measured in different 
units using Z-standardization

Objectives

1 - Create the CustomLogisticRegression class
2 - Create the __init__ method
3 - Create the sigmoid method
4 - Create the predict_proba method
5 - Load the Breast Cancer Wisconsin dataset. Select worst concave points and 
worst perimeter as features and target as the target variable
6 - Standardize X
7 - Split the dataset including the target variable into training and test sets. 
Set train_size=0.8 and random_state=43.

"""

"""Stage 2: Gradient descent with MSE

Description

In this stage, we need to estimate the coef_ (weight) values by gradient descent 
on the Mean squared error cost function. Gradient descent is an optimization 
technique for finding the local minimum of a cost function by first-order 
differentiating. To be precise, we're going to implement the Stochastic 
gradient descent (SGD).

Objectives

1 - Implement the fit_mse method
2 - Implement the predict method

"""

"""Stage 3: Log-Loss

Description

The Mean squared error cost function produces a non-convex graph with the local 
and global minimums when applied to a sigmoid function. If a weight value is 
close to a local minimum, gradient descent minimizes the cost function by the 
local (not global) minimum. This presents grave limitations to the Mean squared 
error cost function if we apply it to binary classification tasks. The Log-loss 
cost function may help to overcome this issue.

Objectives

Implement the fit_log_loss method in class CustomLogisticRegression

"""

# Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.data[['worst concave points', 'worst perimeter', 'worst radius']]
y = data.target

# Standardize X
for feature in X.columns.tolist():
    feature_mean = X[feature].mean()
    feature_std = X[feature].std()
    X[feature] = (X[feature] - feature_mean) / feature_std

# Split the datasets to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    random_state=43)


class CustomLogisticRegression:
    """A simple logistic regression model."""

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = None

    def sigmoid(self, t):
        """The logistic function used to transform the linear regression to
        logistic regression is the sigmoid function."""
        return 1 / (1 + np.exp(- t))

    def predict_proba(self, row, coef_):
        """Predict the probability that <row> belongs to Class 1, given the
        weights <coef_> for each feature."""
        if self.fit_intercept:
            t = np.dot(row, coef_[1:]) + np.array([coef_[0]])
        else:
            t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        """Update the <self.coef_> attribute by estimating the optimal weight
        values using the Gradient Descent method with the Mean Squared Error
        cost function.

        We will start with all weight values being equal to zero.

        """
        if self.fit_intercept:
            count = len(X_train.columns.tolist()) + 1
        else:
            count = len(X_train.columns.tolist())

        # Initialize the weights
        coef_ = np.zeros(count)

        # Training loop
        for _ in range(self.n_epoch):
            i = 0
            for _, row in X_train.iterrows():
                y_hat = self.predict_proba(row, coef_)
                # Update all weights
                if self.fit_intercept:
                    ind = 1
                    for value in row:
                        coef_[ind] = coef_[ind] - self.l_rate * (
                                        y_hat - y_train.iloc[i]) * y_hat * (
                                                     1 - y_hat) * value
                        ind = ind + 1
                    coef_[0] = coef_[0] - self.l_rate * (
                                    y_hat - y_train.iloc[i]) * y_hat * (
                                               1 - y_hat)
                else:
                    ind = 0
                    for value in row:
                        coef_[ind] = coef_[ind] - self.l_rate * (
                                        y_hat - y_train.iloc[i]) * y_hat * (
                                                     1 - y_hat) * value
                        ind = ind + 1
                i = i + 1

        self.coef_ = coef_

    def fit_log_loss(self, X_train, y_train):

        if self.fit_intercept:
            count = len(X_train.columns.tolist()) + 1
        else:
            count = len(X_train.columns.tolist())

        # Initialize the weights
        coef_ = np.zeros(count)

        # Determine number of rows
        N = len(X_train)

        # Training loop
        for _ in range(self.n_epoch):
            i = 0
            for _, row in X_train.iterrows():
                y_hat = self.predict_proba(row, coef_)
                # Update all weights
                if self.fit_intercept:
                    ind = 1
                    for value in row:
                        coef_[ind] = coef_[ind] - (self.l_rate * (
                                        y_hat - y_train.iloc[i]) * value) / N
                        ind = ind + 1
                    coef_[0] = coef_[0] - (self.l_rate * (
                                    y_hat - y_train.iloc[i])) / N
                else:
                    ind = 0
                    for value in row:
                        coef_[ind] = coef_[ind] - (self.l_rate * (
                                        y_hat - y_train.iloc[i]) * value) / N
                        ind = ind + 1
                i = i + 1

        self.coef_ = coef_

    def predict(self, X_test, cut_off=0.5):
        """After the optimal weight values have been determined using the
        self.fit_mse() method and the <self.coef_> has been updates, output the
        prediction of whether the dataset belongs to class 1 or class 0.
        Predictions can only take two values: 0 or 1.

        """
        predictions = self.predict_proba(X_test.to_numpy(), self.coef_)
        predictions[predictions >= cut_off] = 1
        predictions[predictions < cut_off] = 0
        return predictions
