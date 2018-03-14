import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

rdg = Ridge(normalize=False, fit_intercept=True, solver='lsqr', alpha=0.0)

rdg.fit(X_train, y_train)

y_pred = rdg.predict(X_test)

test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",", header=test_header, comments="")