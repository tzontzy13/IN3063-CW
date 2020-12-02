import numpy as np
from mnist_data import get_mnist_data

from MLP import MLP
from MLP_SGD import MLP_SGD

# get data from mnist_data file
test_data, test_targets, train_data, train_targets = get_mnist_data()

# initialize a Stochaistic Gradient Descent Multi Layer Perceptron
# mlp_sgd = MLP_SGD(lr=0.01, sizes=[784,40,30,20,10])
mlp_sgd = MLP_SGD(lr=0.01, sizes=[784, 30, 10])
outputs1, _, _ = mlp_sgd.forward(test_data)
print("Before update: ", mlp_sgd.loss(outputs1, test_targets))
mlp_sgd.fit(train_data, train_targets, 5, 1000)
outputs2, _, _ = mlp_sgd.forward(test_data)
print("After update: ", mlp_sgd.loss(outputs2, test_targets))
# update the weights and biases
# test = mlp_sgd.update([test_data[0]], [test_targets[0]])
