import costFunction

def gradientDes(X, y, theta1, theta2, alpha, num_iters, lmbda, input_layer, hidden_layer, num_labels):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        cost, grad1, grad2 = costFunction.cost(theta1, theta2, input_layer, hidden_layer, num_labels, X, y, lmbda)
        theta1 = theta1 - (alpha * grad1)
        theta2 = theta2 - (alpha * grad2)
        J_history.append(cost)

    return theta1, theta2, J_history;