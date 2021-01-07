import numpy as np
import pandas as pd
import sigmoid
import csv
from numpy import array, savetxt

def prediction(theta1, theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))

    a1 = sigmoid.sig(X @ theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))  # hidden layer
    a2 = sigmoid.sig(a1 @ theta2.T)  # output laye
    ans = np.argmax(a2, axis=1)

    return np.argmax(a2, axis=1)