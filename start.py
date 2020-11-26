import numpy as np
from mnist_data import get_mnist_data

from MLP import MLP
from MLP_SGD import MLP_SGD

# train_data, train_targets, test_data, test_targets = get_mnist_data()

test_data, test_targets = get_mnist_data()

mlp = MLP([784, 30, 10])

# test = mlp.forward(test_data)
a, b, x = mlp.forward2(test_data)
# loss = mlp.loss(test_targets, output)
print(len(x))
