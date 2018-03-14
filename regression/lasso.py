import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Lasso

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

X_train = np.delete(X_train, [0,2,4,5], axis=1)
X_test = np.delete(X_test, [0,2,4,5], axis=1)

lasso = Lasso(normalize=True, alpha=0.3, fit_intercept=True)

lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",", header=test_header, comments="")