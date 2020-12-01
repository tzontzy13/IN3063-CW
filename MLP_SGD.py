from MLP import MLP

import numpy as np


class MLP_SGD(MLP):

    # initialize SGD with learning rate and sizes ( [784, 30, 10] - example )
    def __init__(self, sizes, lr=0.01):

        super().__init__(sizes)

        self.lr = lr

    # this is where the program "starts"
    # pass training data and start iterating through images
    def fit(self):
        # parameters are images, LEARNING RATE, MINI_BATCH_SIZE
        # split TRAINING DATA in MINI BATCHES
        # for each minibatch, call update
        # returns gradient for each weight and bias
        # average the gradients
        # update weights and biases with formula
        # new_weight -= learning_rate * gradient
        pass

    # calculate the gradient of all images in a minibatch
    # CHANGE NAME OF THIS FUNCTION
    def update(self, X_train, y_train):

        # for each image in X_train ( X_train size = minibatch size)
        for x, y in zip(X_train,y_train):

            # array of zeros so we can use vector notation
            gradient_b = [np.zeros(b.shape) for b in self.biases]
            gradient_w = [np.zeros(w.shape) for w in self.weights]
            
            # get activations and Z's OF THIS IMAGE from forward
            activations, zs = self.forward3(x)

            # ONLY FOR LAST LAYER

            # derivative of cost depending on Z's
            # "CHAIN RULE"
            # derivative of cost depending on output = first element below
            # derivative of (output - y_pred) ^ 2 = 1/2 * (output - y_pred) so, output - y_pred
            # derivative of output depending on Z's = second element below
            sigma_z = (activations[-1] - y) * self.sigmoid_derivated(zs[-1])

            # bias gradient is sigma_z * derivative of Z's depending on bias, ALWAYS = 1
            # because z = w0 * a0 + w1 * a1 + ... + wn * an + b
            gradient_b[-1] = sigma_z
            # weight gradient is sigma_z * derivative of Z's depending on weight, which is activation
            # i use transpose because i need the shapes to match
            gradient_w[-1] = sigma_z * activations[-2].transpose()

            # for ALL OTHER LAYERS

            for i in range(self.num_layers-2, 0, -1):
                print("\n")
                print("total number of layers: ", self.num_layers)
                print("LAYER NUMBER: ", i)
                print("Z's shape:         ", zs[i-1].shape)
                print("z derivated shape: ", self.sigmoid_derivated(zs[i-1]).shape)
                print("sigma_z shape:     ", sigma_z.shape)
                print("weights shape:     ", self.weights[i-1].shape)
                print('\n')
                # sigma_z = self.sigmoid_derivated(zs[i]) * sigma_z * self.weights[i]
                # gradient_b[i] = sigma_z
                # gradient_w[i] = sigma_z * activations[i-1]
                pass