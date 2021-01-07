import pandas as pd
import sys
import numpy as np
import costFunction
import gradientDescent
import predict

# read data from file
data = pd.read_csv("train.csv")
y = data["label"].values    # 42000 * 1
X = data.drop(["label"], axis=1).values   # 42000 * 784

m = X.shape[0]

# declarign variables
input_layer = X.shape[1]   #784
hidden_layer = 25
num_labels = 10   #10
alpha = 0.8
num_iters = 8
lmbda = 3

# creating initial theta1 and theta2 for the neural network value
theta1 = np.random.rand(hidden_layer, input_layer+1) * 2 * sys.float_info.epsilon - sys.float_info.epsilon    # 25 * 785
theta2 = np.random.rand(num_labels, hidden_layer+1) * 2 * sys.float_info.epsilon - sys.float_info.epsilon   # 10 * 26

theta11, theta22, J = gradientDescent.gradientDes(X, y, theta1, theta2, alpha, num_iters, lmbda, input_layer, hidden_layer, num_labels)

#test_data = pd.read_csv("test.csv")
pred3 = predict.prediction(theta11, theta22, X)
print("Training Set Accuracy:",sum(pred3[:,np.newaxis]==y)[0]/m*100,"%")