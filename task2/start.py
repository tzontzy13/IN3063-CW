from get_data import load_dataset
from Network import Network
import numpy as np

# you change the configuration for the network in the init of Network class
print('\n')
network = Network()
print("Network initialized")
print('\n')

# retrieve data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
print("Data has been loaded")
print("Number of images in dataset: ", len(X_train) + len(X_val))
print("Number of images to test on: ", len(X_test))
print('\n')

# parameters for the network
epochs = 5000
minibatch_length = 32

print("Number of epochs: ", epochs)
print("Minibatch length: ", minibatch_length)
print('\n')

print("Epochs have started!")
print('\n')

# initial values for loss for each batch and accuracy
loss_list_on_batches = []
acc = 0

# start program here
# for each epoch
for epoch in range(epochs):

    print("Current epoch: ", epoch + 1)

    # split data in minibatches
    X_batches, y_batches = network.split_data_in_batches(
        X_train, y_train, minibatch_length=minibatch_length)

    # for each minibatch
    for X_batch, y_batch in zip(X_batches, y_batches):
        # calculate batch loss
        batch_loss, grad_for_stopping_criterion = network.train(
            X_batch, y_batch)
        # append minibatch loss to list of all losses
        loss_list_on_batches.append(batch_loss)

    # calculate accuracy
    acc_sum = network.accuracy(X_val, y_val)

    # STOPPING CRITERION
    # we decided to have the loss as our stopping criterion
    # our loss can go up or down because the Categorical Cross-Entropy loss (WE DID OURSELVES) and gradient for our loss (WHICH WE COPIED - reference in file)
    # are similar, but not dependent on each other, meaning:
    # we did not use the Categorical Cross-Entropy loss funtion to compute the gradient
    # BUT, it still works, the cost we calculated gets lower with each minibatch so we can compare it with the last minibatch
    # if it doesnt change much, we stop training and print accuracy and loss
    if(np.abs(loss_list_on_batches[-1] - loss_list_on_batches[-1 - minibatch_length]) < 0.005):

        print("Accuracy:      ", acc_sum, "/", len(y_val))
        print("Loss:          ", loss_list_on_batches[-1])
        print('\n')
        print("--- NN has saturated ---")
        break

    # if we havent reached saturation, we print this at the end of the last epoch
    print("Accuracy:      ", acc_sum, "/", len(y_val))
    print("Loss:          ", loss_list_on_batches[-1])
    print('\n')
#     plt.plot(train_log,label='train accuracy')
#     plt.plot(val_log,label='val accuracy')
#     plt.legend(loc='best')
#     plt.grid()
#     plt.show()

# here we test on X_TEST and Y_TEST
# we return the accuracy on the TEST DATA
print("Testing")
test_acc = network.accuracy(X_test, y_test)
print("Test accuracy: ", test_acc, "/", len(y_test))
