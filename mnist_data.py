import numpy as np
import torch

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# looked for documentation on
# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
# https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
# downloaded both test and train sets

# transforms the data to Tensor format
transform = transforms.Compose(
    [transforms.ToTensor()])

# download data from torchvision
trainset = datasets.MNIST(root='./data',
                          train=True,
                          transform=transform,
                          download=True,
                          )

testset = datasets.MNIST(root='./data',
                         train=False,
                         transform=transform,
                         download=True,
                         )

# transform a number-target into a matrix, for use in loss function
# for example 5 -> [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
# used for easier calculation of  loss function
def y_to_vector(n):

    # create an matrix of zeros then set a value of 1 to the index of input number
    x = np.zeros((10,1))
    x[n] = 1

    return x

# method to transform the data for the desired inputs and outputs
def get_mnist_data():

    # train_data = trainset.data.numpy()
    # train_data = train_data.reshape(60000, 784, 1)
    # train_data = np.array([train_data[i]/max(train_data[i])
    #                        for i in range(len(train_data))])

    # train_targets = trainset.targets.numpy()
    # train_targets = train_targets.reshape(60000, 1)
    # train_targets = np.array([y_to_vector(train_targets[i])
    #                           for i in range(len(train_targets))])

    test_data = testset.data.numpy()
    test_data = test_data.reshape(10000, 784, 1)
    test_data = np.array([test_data[i]/max(test_data[i])
                          for i in range(len(test_data))])

    test_targets = testset.targets.numpy()
    test_targets = test_targets.reshape(10000, 1)
    test_targets = np.array([y_to_vector(test_targets[i])
                             for i in range(len(test_targets))])

    # return the 4 data structures
    return test_data, test_targets
    #train_data, train_targets