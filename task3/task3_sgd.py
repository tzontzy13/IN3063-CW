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

# Hyper params
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.05

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.soft = nn.Softmax()

        # self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(),
                      lr=learning_rate)

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

    print('Epoch {}'.format(epoch))


loss_list_on_epochs = [0]
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
