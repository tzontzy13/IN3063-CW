import numpy as np
from mnist_data import get_mnist_data

from MLP import MLP
from MLP_SGD import MLP_SGD

# get data from mnist_data file
test_data, test_targets, train_data, train_targets = get_mnist_data()

# initialize a Stochaistic Gradient Descent Multi Layer Perceptron

mlp_sgd = MLP_SGD(lr=1, sizes=[784, 30, 16, 10],
                  activation_list=['sigmoid', 'softmax'])
outputs1 = mlp_sgd.forward(test_data)
print("Before update: ", mlp_sgd.loss(outputs1, test_targets))
print("Accuracy: ", mlp_sgd.evaluate(test_data, test_targets))

mlp_sgd.fit(train_data, train_targets, 5, 10)

outputs2 = mlp_sgd.forward(test_data)
print("Loss after update: ", mlp_sgd.loss(outputs2, test_targets))
print("Accuracy: ", mlp_sgd.evaluate(test_data, test_targets))
