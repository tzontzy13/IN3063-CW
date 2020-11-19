import random
import numpy as np


class MLP(sizes):

    def __init__(self, sizes):
        self.sizes = sizes
        # [784, 30, 10]
        self.biases = []
        self.weights = []

        for i in range(1, len(sizes)):
            self.biases.append(np.random.randn(sizes[i]))

        for i, j in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(j, i))

    def forward(self, x):
        a = []

        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
            a.append(x)

        return a

    # def loss(y, a):
    #     pass

    # def accuracy():
    #     pass

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
