#!/usr/bin/env python

import numpy as np


# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]


# Fit model and predict test values
y_pred = np.random.randint(y_train.min(), y_train.max(), X_test.shape[0])

# X(^T)X

temp = np.dot(np.transpose(X_train), X_train)

# (X(^T)X)^-1

temp = np.linalg.inv(temp)

# ((X(^T)X)^-1)X(^T)

temp = np.dot(temp, np.transpose(X_train))

# w = ((X(^T)X)^-1)X(^T)t

w = np.dot(temp, y_train)

# prediction = wx
y_pred = np.dot(w, np.transpose(X_test))

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.