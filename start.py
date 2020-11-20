import numpy as np
from mnist_data import get_mnist_data

from MLP import MLP

train_data, train_targets, test_data, test_targets = get_mnist_data()

mlp = MLP([784,30,10])

#print(len(train_data[0]))
x = mlp.forward(test_data[0])
#y = mlp.forward(x)

print(x[2])
