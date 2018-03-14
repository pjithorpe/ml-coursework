import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# fix random seed for reproducibility
np.random.seed(7)

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

print(X_train)
# fully connect 2 layers
def create_model():
	m = Sequential()
	m.add(Dense(20, input_dim=112, kernel_initializer='normal', activation='relu'))
	m.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

	m.compile(loss='binary_crossentropy', optimizer='adam')

	return m

estimator = KerasClassifier(build_fn=create_model, verbose=10)

# Train the model using the training sets
estimator.fit(X_train, y_train, epochs=100, batch_size=5)

# Make predictions using the testing set
y_pred = estimator.predict(X_test)
y_pred = np.hstack(y_pred)
print(y_pred)

test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",", header=test_header, comments="")