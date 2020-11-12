import torch

import torchvision
import torchvision.datasets as datasets

# looked for documentation on
# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
# https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision
# downloaded both test and train sets

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True)

print(len(mnist_trainset))
print(len(mnist_testset))
