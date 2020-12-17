import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn
import torchvision
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Hyper params sett
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./task3data', train=True, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor()
#                                ])),
#     batch_size=batch_size_train, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./task3data', train=False, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor()
#                                ])),
#     batch_size=batch_size_test, shuffle=True)

trainset = torchvision.datasets.MNIST('./task3data', train=True, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor()
                                      ]))

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_train, shuffle=True)

testset = torchvision.datasets.MNIST('./task3data', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor()
                                     ]))
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size_test, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.soft = nn.Softmax()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(),
                      lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = network(data)

        # backward pass
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()

        # update params
        optimizer.step()

        if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

    print('Epoch {} Loss: {:.6f}'.format(
        epoch, train_losses[-1]))


loss_list_on_epochs = []
acc_list_on_epochs = []


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target,
                                    size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    loss_list_on_epochs.append(test_loss)
    acc_list_on_epochs.append(100. * correct / len(test_loader.dataset))
    # test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
    if abs(loss_list_on_epochs[-1] - loss_list_on_epochs[-2]) < 0.005:
        print("Network Saturated")
        loss_list_on_epochs = loss_list_on_epochs[1:]
        break

# Source: https://deeplizard.com/learn/video/0LhiS6yu2qQ


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds), dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels), dim=0
        )
    return all_preds, all_labels


y_pred, y_test = get_all_preds(network, test_loader)
y_pred = np.argmax(y_pred, axis=1)

# PLOTS

plt.plot(loss_list_on_epochs)
plt.title("loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='best')
plt.show()

plt.plot(acc_list_on_epochs)
plt.title("accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'], loc='best')
plt.show()

cm = confusion_matrix(y_test, y_pred)
ax = plt.subplot()
ax.set_title('Predicted vs Actual')
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sns.heatmap(cm, annot=True, ax=ax, cmap='Reds', fmt='g')
plt.xlabel('Predicted labels', axes=ax)
plt.ylabel('True labels', axes=ax)
plt.show()
