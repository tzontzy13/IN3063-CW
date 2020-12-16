from CrossEntropy import CrossEntropy
from Dense import Dense
from Relu import Relu
from Sigmoid import Sigmoid
import numpy as np

# import matplotlib.pyplot as plt

# Reference: Aayush Agrawal. 2020
# Building Neural Network from scratch | by Aayush Agrawal | Towards Data Science.
# [ONLINE] Available at: https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
# [Accessed 15 December 2020]


class Network():

    def __init__(self):
        self.network = []

        # this is where you "build" your network structure
        # Adds the sizes and types of layers used in the network
        self.network.append(Dense(784, 200))
        self.network.append(Relu())
        self.network.append(Dense(200, 100))
        self.network.append(Relu())
        self.network.append(Dense(100, 10))

        self.cross_entropy = CrossEntropy()

    # Returns activations for all layers.
    def forward(self, image_input):

        prev_layer_activations = image_input
        all_activations = []

        # Performs the forward step by going through each layer's forward function
        # Afterwards, stores the activation for the next iteration in the next layer
        for layer in self.network:
            # uses each layers forward function (example: relu and dense have a different one)
            next_activation = layer.forward(prev_layer_activations)
            all_activations.append(next_activation)
            # set for next iteration
            prev_layer_activations = all_activations[-1]

        return all_activations

    # Calculates the accuracy of the network by summing up the right-guessed answers
    # It makes use of the output of the last activation and the targets
    def accuracy(self, images, targets):

        outputs = self.forward(images)

        # stores the network output
        y_actual = outputs[-1]

        y_pred = []

        for n in y_actual:
            number = np.argmax(n)
            y_pred.append(number)

        acc_sum = 0
        for pred, actual in zip(y_pred, targets):
            if pred == actual:
                acc_sum += 1

        return acc_sum

    # Trains the network by performing forward and backward pass on each mini-batch
    def train(self, images, targets):

        # Perform the forward pass on the given data. The last layer of the activations returns
        # the Zs without any activation applied to it as it will be applied below when calculating the cost
        activations = self.forward(images)
        final_zs = activations[-1]
        # Adds the mini-batch image inputs to be used later
        all_inputs = [images] + activations

        # Calculates the gradient of the last layer, at each neuron
        # function from CrossEntropy class
        gradient_l = self.cross_entropy.grad_softmax_crossentropy_with_logits(
            final_zs, targets)

        # This is where the backward pass starts.
        # prepare inputs for backpropagation
        # cant use np.flip because activations is an ndarray with multiple dimensions
        flip_inputs = []
        for i in range(len(activations) - 1, -1, -1):
            flip_inputs.append(all_inputs[i])

        # propagates the error on all layers
        # using np.flip because here self.network is just a list of length (5)
        for layer, input in zip(np.flip(self.network), flip_inputs):
            # "go backwards" for each layer, calculating a new gradient dor that layer
            gradient_l = layer.backward(input, gradient_l)

    def loss(self, images, targets):
        activations = self.forward(images)
        final_zs = activations[-1]

        all_losses = self.cross_entropy.softmax_cross_entropy_loss(
            final_zs, targets)
        # median of all losses for each image
        loss = np.sum(all_losses / len(all_losses))

        return loss
        
    # function for splitting dataset in minibatches
    def split_data_in_batches(self, images, targets, minibatch_length):

        # data structures for all images and targets
        X_batches = []
        y_batches = []

        # shuffle starts here
        # combine images with targets
        mix = []
        for image, target in zip(images, targets):
            mix.append([image, target])

        # shuflle mix
        np.random.shuffle(mix)
        mix2 = np.array(mix, dtype=object)

        # data structures for shuffled images and targets
        images3 = []
        targets3 = []

        # add shuffled images in targets to appropiate data structure
        for i, t in mix2:
            images3.append(i)
            targets3.append(t)

        # transform data so you have the same type as initial data passed to function
        images3 = np.asarray(images3, dtype='float64')
        targets3 = np.array(targets3, dtype='uint8')
        # images3 and targets3 are shuffled data

        #  append to minibach lists
        for i in range(0, len(images), minibatch_length):

            test_x = images3[i: i + minibatch_length]
            test_y = targets3[i: i + minibatch_length]
            X_batches.append(test_x)
            y_batches.append(test_y)

        return X_batches, y_batches
