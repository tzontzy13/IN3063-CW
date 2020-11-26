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

        self.biases = np.array(self.biases, dtype=object)
        self.weights = np.array(self.weights, dtype=object)

    def forward(self, xs):

        x_layers = []

        for x in xs:
            all_layer_activations = []

            inputs = x
            for i in range(len(self.weights)):
                single_layer_activations = []

                for b, w in zip(self.biases[i], self.weights[i]):
                    single_layer_activations.append(
                        self.sigmoid(np.dot(w, inputs)+b))

                all_layer_activations.append(single_layer_activations)
                inputs = single_layer_activations

            x_layers.append(all_layer_activations)

        return x_layers

    def forward2(self, x):

        #print(x[0])
        print(self.weights[0].shape)
        # print(self.weights[1].shape)
        # print(self.biases.shape)

        # [1 2 3 4 5 ... 784]
        # [1 2 3 4 5 ... 784]
        # [1 2 3 4 5 ... 784]
        # .
        # .
        # . 30 de ori
        # .
        # [1 2 3 4 5 ... 784]

        # activation_hidden_0 = self.sigmoid(x[0][0] * self.weights[0][0][0] + x[0][1] * self.weights[0][0][1]+ self.biases[0][0])
        # activation_hidden_1 = self.sigmoid(x[:,0] * self.weights[0][0][0] + self.biases[0][0])
        
        #layer1_weights = self.weights[0]

        #activation_hidden_2 = self.sigmoid(x[:,0] * self.weights[0][0][0] + x[:,1] * self.weights[0][0][1])

    def loss(self, a, y):
        return 0.5*np.sum((y-a)^2)

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
