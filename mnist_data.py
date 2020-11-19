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
                               # transform=transforms.Compose([
                               #     transforms.ToTensor()
                               # ])
                               )

def get_mnist_data():

    train = trainset
    #validation = trainset
    test = testset
    
    return train, test