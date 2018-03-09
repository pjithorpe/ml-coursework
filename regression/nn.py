import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
# fix random seed for reproducibility
np.random.seed(7)

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

print(X_train)
# fully connect 2 layers
m = Sequential()
m.add(Dense(20, input_dim=6, kernel_initializer='normal', activation='relu'))
m.add(Dense(1, kernel_initializer='normal'))

m.compile(loss='mean_squared_error', optimizer='adam')

# Train the model using the training sets
m.fit(X_train, y_train, epochs=500, batch_size=1)

# Make predictions using the testing set
y_pred = m.predict(X_test)
y_pred = np.hstack(y_pred)
print(y_pred)

test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",", header=test_header, comments="")