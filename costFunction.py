import numpy as np
import sigmoid
import math

def cost(theta1, theta2, input_layer, hidden_layer, num_labels, X, y, lmbda):
    J = 0
    m = len(X)  # 42000


    theta1_grad = np.zeros([hidden_layer, input_layer+1])   # 25 * 785
    theta2_grad = np.zeros([num_labels, hidden_layer+1])    # 10 26

    # calculating cost
    # adding a column of 1
    X = np.hstack((np.ones((m, 1)), X))

    a1 = X      # 42000 * 785
    z2 = a1 @ theta1.T    # 42000 * 25
    a2 = sigmoid.sig(z2)
    # adding a column of 1
    a2 = np.hstack((np.ones((m, 1)), a2))   # 42000 * 26
    z3 = a2 @ theta2.T    # 42000 * 10
    a3 = sigmoid.sig(z3)    # 42000 * 10
    hx = a3

    # making y vector
    y_vec = np.zeros([m, num_labels])
    for i in range(m):
        y_vec[i][y[i]] = 1

    for i in range(m):
        for k in range(num_labels):
            J = J + y_vec[i][k]*math.log(hx[i][k]) + (1-y_vec[i][k])*math.log(1-hx[i][k])

    J = J * (-1.0/m)

    # calculating delta
    delta3 = a3 - y_vec     # 42000 * 10
    delta2 = delta3 @ theta2     # 42000 * 26
    for i in range(m):
        for k in range(hidden_layer+1):
            delta2[i][k] = delta2[i][k] * a2[i][k] * (1-a2[i][k])
    delta2 = delta2[0:m, 1:hidden_layer+1]    # 42000 * 25

    theta1_grad = (1.0/m) * (delta2.T @ a1)    # 25 * 785
    theta2_grad = (1.0 / m) * (delta3.T @ a2)  # 10 * 26

    # calculating Regularization term
    reg_term = 0
    for i in range(hidden_layer):
        for j in range(1, input_layer+1):
            reg_term = reg_term + theta1[i][j]*theta1[i][j]

    for i in range(num_labels):
        for j in range(1, hidden_layer+1):
            reg_term = reg_term + theta2[i][j]*theta2[i][j]

    reg_term = reg_term * (lmbda/(2*m))
    J = J + reg_term

    for i in range(hidden_layer):
        for j in range(1, input_layer+1):
            theta1_grad[i][j] = theta1_grad[i][j] + (lmbda/m)*theta1_grad[i][j]

    for i in range(num_labels):
        for j in range(1, hidden_layer+1):
            theta2_grad[i][j] = theta2_grad[i][j] + (lmbda/m)*theta2_grad[i][j]


    return J, theta1_grad, theta2_grad;