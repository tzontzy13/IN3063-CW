import numpy as np
import torch

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# looked for documentation on
# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
# https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
# downloaded both test and train sets

trainset = datasets.MNIST(root='./data',
                          train=True,
                          download=True,
                          )

testset = datasets.MNIST(root='./data',
                         train=False,
                         download=True,
                         )


def y_to_vector(n):
    # transform a number-target into a matrix, for use in loss function

    # create an matrix of zeros then set a value of 1 to the index of input number
    x = np.zeros((10, 1))
    x[n] = 1

    return x


def get_mnist_data():

    # created data structures for both training and testing sets
    # split them into data and targets

    train_data = []
    train_targets = []

    test_data = []
    test_targets = []

    # reshape a 28x28 matrix into a NORMALIZED 784 vector
    for i in trainset.data:
        train_data.append(i.reshape(784)/255)

    # transformed the target from number to vector, for use in loss function
    for i in trainset.targets:
        train_targets.append(y_to_vector(i))

    # reshape a 28x28 matrix into a NORMALIZED 784 vector
    for i in testset.data:
        test_data.append(i.reshape(784)/255)

    # transformed the target from number to vector, for use in loss function
    for i in testset.targets:
        test_targets.append(y_to_vector(i))

    # return the 4 data structures
    return train_data, train_targets, test_data, test_targets