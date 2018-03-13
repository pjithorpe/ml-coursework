import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

lasso = Lasso()
alphas = np.logspace(-5, 5, 80)

params_grid = [
	{
		'selection': ['cyclic', 'random'],
		'positive': [True, False],
		'max_iter': [50, 100, 1000, 5000, 10000, 100000, 1000000],
		'normalize': [True, False],
		'fit_intercept': [True, False],
		'alpha': [0.0, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2, 4, 6, 10]
	}
]

gridsearch = GridSearchCV(lasso, params_grid, n_jobs=-1, verbose=True)
gridsearch.fit(X_train, y_train)

best_params = gridsearch.best_params_
print(best_params)
y_pred = gridsearch.predict(X_test)

test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",", header=test_header, comments="")