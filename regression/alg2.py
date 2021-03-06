import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=True, normalize=True)

lr_pipeline = Pipeline([
	('lr', linear_model.LinearRegression()),
])

params_grid = [
	{
		'lr__fit_intercept': [True, False],
		'lr__normalize': [True, False],
	}
]

lr_grid_search = GridSearchCV(lr_pipeline, params_grid, n_jobs=-1, verbose=1)
print("LinearRegression: \n")
lr_grid_search.fit(X_train, y_train)

print(lr_grid_search.best_params_)
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",", header=test_header, comments="")