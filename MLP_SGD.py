from MLP import MLP, sigmoid_derivated

import numpy as np


class MLP_SGD(MLP):

    # initialize SGD with learning rate and sizes ( [784, 30, 10] - example )
    def __init__(self, sizes, lr=0.01):

        super().__init__(sizes)

        self.lr = lr

    # this is where the program "starts"
    # pass training data and start iterating through images
    def fit(self, X_train, y_train, epochs, mini_batch_size):

        # split TRAINING DATA in MINI BATCHES
        X_mini_batches = [X_train[x:x+mini_batch_size]
                          for x in range(0, len(X_train), mini_batch_size)]
        y_mini_batches = [y_train[y:y+mini_batch_size]
                          for y in range(0, len(y_train), mini_batch_size)]

        for i in range(epochs):
            print("Current epoch: ", i)
            # for each minibatch, call update
            for mini_X, mini_y in zip(X_mini_batches, y_mini_batches):
                # the update function returns the mean of the gradients
                # for the weight and bias
                mini_weights, mini_biases = self.update(mini_X, mini_y)
                # update the weights and biases using the right formula
                self.weights = [w-self.lr*mini_w
                                for w, mini_w in zip(self.weights, mini_weights)]
                self.biases = [b-self.lr*mini_b
                               for b, mini_b in zip(self.biases, mini_biases)]

        print("done")

    # calculate the gradient of all images in a minibatch
    def update(self, X_train, y_train):

        # init with zeros so i can use vector notation when calculation the average
        all_gradient_b = []
        all_gradient_w = []

        # for each image in X_train ( X_train size = minibatch size)
        for x, y in zip(X_train, y_train):

            # array of zeros so we can use vector notation
            gradient_b = [np.zeros(b.shape) for b in self.biases]
            gradient_w = [np.zeros(w.shape) for w in self.weights]

            # get activations and Z's OF THIS IMAGE from forward
            activations, zs = self.forward2(x)

            # ONLY FOR LAST LAYER

            # derivative of cost depending on Z's
            # "CHAIN RULE"
            # derivative of cost depending on output = first element below
            # derivative of (output - y_pred) ^ 2 = 1/2 * (output - y_pred) so, output - y_pred
            # derivative of output depending on Z's = second element below

            sigma_z = (activations[-1] - y) * sigmoid_derivated(zs[-1])

            # bias gradient is sigma_z * derivative of Z's depending on bias, ALWAYS = 1
            # because z = w0 * a0 + w1 * a1 + ... + wn * an + b
            gradient_b[-1] = sigma_z / len(X_train)
            # weight gradient is sigma_z * derivative of Z's depending on weight, which is activation
            # i use transpose because i need the shapes to match
            gradient_w[-1] = sigma_z * activations[-2].transpose()
            gradient_w[-1] = gradient_w[-1] / len(X_train)

        return self.weights, self.biases
