import random
import numpy as np


class MLP():

    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = []
        self.weights = []

        for i in range(1, len(sizes)):
            self.biases.append(np.random.randn(sizes[i]))

        for i, j in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(j, i))

    def forward(self, x):
        all_layer_activations = []
        inputs = x

        for i in range(len(self.weights)):
            single_layer_activations = []

            for b, w in zip(self.biases[i], self.weights[i]):
                single_layer_activations.append(self.sigmoid(np.dot(w, inputs)+b))

            all_layer_activations.append(single_layer_activations)
            inputs = single_layer_activations

        return all_layer_activations

    def loss(self, y, a):
        pass

    def accuracy(self):
        pass

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def sigmoid_derivated(n):
        return sigmoid(n)*(1-sigmoid(n))

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def relu_derivative(x):
        if x > 0:
            return 1
        else:
            return 0
