import torchvision
import torch


# Reference 1 — PyTorch 1.7.0 documentation. 2020.
# torchvision.datasets — PyTorch 1.7.0 documentation.
# [ONLINE] Available at: https://pytorch.org/docs/stable/torchvision/datasets.html#mnist.
# [Accessed 17 December 2020].

# Reference 2 - aiworkbox.com. 2020.
# PyTorch MNIST: Load MNIST Dataset from PyTorch Torchvision · PyTorch Tutorial .
# [ONLINE] Available at: https://www.aiworkbox.com/lessons/load-mnist-dataset-from-pytorch-torchvision.
# [Accessed 17 December 2020].


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
