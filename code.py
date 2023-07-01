# Imported packages
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

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


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.exp(- t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = np.dot(row, coef_[1:]) + np.array([coef_[0]])
        else:
            t = np.dot(row, coef_)
        return self.sigmoid(t)


# Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.data[['worst concave points', 'worst perimeter']]
y = data.target

# Standardize X
for feature in ['worst concave points', 'worst perimeter']:
    feature_mean = X[feature].mean()
    feature_std = X[feature].std()
    X[feature] = (X[feature] - feature_mean) / feature_std

# Split the datasets to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    random_state=43)

