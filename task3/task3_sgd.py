import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from get_data import load_dataset

# Hyper params
n_epochs = 10
batch_size_train = 35
batch_size_test = 1000
learning_rate = 0.05
# Retrieve the data
train_loader, test_loader = load_dataset(batch_size_train, batch_size_test)


# References for the main idea of building the __init__/forward functions:
# Training a Classifier — PyTorch Tutorials 1.7.1 documentation. 2020.
# [ONLINE] Available at: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
# [Accessed 17 December 2020].
# MNIST Handwritten Digit Recognition in PyTorch - Nextjournal. 2020.
# [ONLINE] Available at: https://nextjournal.com/gkoehler/pytorch-mnist
# [Accessed 17 December 2020].
class SGD(nn.Module):
    def __init__(self):
        super(SGD, self).__init__()
        # Set up the structure of layers for the network
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        # Call the activation functions of each layer in order
        # Resizes/ Flattens the input data
        # to be usable in the first activation function
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


# Initialize the network and apply the SGD optimizeer
network = SGD()
optimizer = optim.SGD(network.parameters(),
                      lr=learning_rate)


def train():
    # Iterate through each mini_batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # get the output from the forward pass
        output = network(data)
        # backward pass
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        # update params
        optimizer.step()


# Lists for usage in the plots section
loss_list_on_epochs = [0]
acc_list_on_epochs = []
total_training_time = []


# Reference for the loss function used:
# torch.nn.functional — PyTorch 1.7.0 documentation.
# [ONLINE] Available at: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#cross_entropy.
# [Accessed 17 December 2020].
# Reference for accuracy:
# torch.max — PyTorch 1.7.0 documentation. 2020.
# [ONLINE] Available at: https://pytorch.org/docs/stable/generated/torch.max.html.
# [Accessed 17 December 2020].
def test(epoch):
    test_loss = 0
    correct = 0
    # iterate through each mini-batch
    for data, target in test_loader:
        # forward pass
        output = network(data)
        # sum the losses for each mini-batch
        test_loss += F.cross_entropy(output, target,
                                     size_average=False).item()
        # generate accuracy
        # We only use the second returned parameter
        _, pred = torch.max(output, 1)
        for pred, actual in zip(pred, target):
            if pred == actual:
                correct += 1
    # get mean loss from diving the sum by the number of batches
    test_loss /= len(test_loader.dataset)
    # Update params with the current loss and accuracy
    loss_list_on_epochs.append(test_loss)
    acc_list_on_epochs.append(100. * correct / len(test_loader.dataset))
    # Print out the statistics for each epoch
    print('Epoch {} finished'.format(epoch))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} \n'.format(
        test_loss, correct, len(test_loader.dataset)))


start = time.time()
# Start iterating through the epochs and call the train and test function for each.
for epoch in range(n_epochs):
    # start the time lapse and then update the total_training_time list for each epoch
    train()
    test(epoch + 1)
    end = time.time()
    total_training_time.append(end - start)
    # Stopping criterion with respect to the loss
    if abs(loss_list_on_epochs[-1] - loss_list_on_epochs[-2]) < 0.003:
        print("Network Saturated")
        loss_list_on_epochs = loss_list_on_epochs[1:]
        break


# Reference: DeepLizard. 2020.
# CNN Confusion Matrix with PyTorch - Neural Network Programming.
# [ONLINE] Available at: https://deeplizard.com/learn/video/0LhiS6yu2qQ.
# [Accessed 17 December 2020].
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


# Retrieves the outputs of the network together with their respective labels
# for use in the confusion matrix below
y_pred, y_test = get_all_preds(network, test_loader)
y_pred = np.argmax(y_pred, axis=1)

# PLOTS
# Reference: Like Geeks. 2020.
# Seaborn heatmap tutorial (Python Data Visualization) - Like Geeks.
# [ONLINE] Available at: https://likegeeks.com/seaborn-heatmap-tutorial/.
# s[Accessed 17 December 2020].
plt.plot(total_training_time)
plt.title("elapsed time")
plt.ylabel('elapsed time')
plt.xlabel('epoch')
plt.legend(['elapsed time'], loc='best')
plt.show()

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
