import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]

svc = LinearSVC()
# Fit model and predict test values

params_grid = [
	{
		'max_iter': [50, 100, 1000, 5000, 10000],
		'dual': [False],
		'C': [0.5, 1.0, 2.0, 10.0, 50.0, 100.0],
		'multi_class': ['ovr', 'crammer_singer'],
		'fit_intercept': [True, False],
	}
]

gridsearch = GridSearchCV(svc, params_grid, n_jobs=-1, verbose=30)
gridsearch.fit(X_train, y_train)

best_params = gridsearch.best_params_
print(best_params)
y_pred = gridsearch.predict(X_test)

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.