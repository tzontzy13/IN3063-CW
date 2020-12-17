import torchvision
import torch

# Referece: Aayush Agrawal. 2020
# Building Neural Network from scratch | by Aayush Agrawal | Towards Data Science.
# [ONLINE] Available at: https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
# [Accessed 15 December 2020]

# get data as in the refference above, under the "Full network" section


def load_dataset(batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./task3data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./task3data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader
