# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import numpy as np
import scipy
import scipy.stats

def kNNClassification(X_train, y_train, X_test, k):
    i=-1
    KNNPredictionMatrix = np.zeros((X_test.shape[0],))
    for X_test_instance in X_test:
        i = i + 1
        nearestNeighbour = np.argsort(np.linalg.norm(X_train - X_test_instance, axis=1))[:k]
        KNNPredictionMatrix[i] = stats.mode(y_train[nearestNeighbour], axis=None).mode[0]
    return KNNPredictionMatrix

	
def main():
	# Prepare the data.
	X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
	X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
	y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]
	X_train_new, X_test_new = X_train[:len(X_train)/2], X_train[len(X_train)/2:]
	y_train_new, y_test_new = y_train[:len(y_train)/2], y_train[len(y_train)/2:]
	k=3
	print(kNNClassification(X_train_new, y_train_new, X_test_new, k))

main()