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

        all_activations_for_all_inputs = []

        for x in xs:

            all_activations_for_one_input = []
            inputs = x

            for i in range(len(self.weights)):

                single_layer_activations = []

                for b, w in zip(self.biases[i], self.weights[i]):

                    single_layer_activations.append(
                        self.sigmoid(np.dot(w, inputs)+b))

                all_activations_for_one_input.append(single_layer_activations)
                inputs = single_layer_activations

            all_activations_for_all_inputs.append(
                all_activations_for_one_input)

        return np.array(all_activations_for_all_inputs, dtype=object)

    def forward2(self, xs):

        x_layers = []
        outputs = []
        hidden = []
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

        for x in x_layers:
            outputs.append(x[1])

        for x in x_layers:
            hidden.append(x[0])

        return np.array(outputs, dtype=object), x_layers

    def forward3(self, x):
        pass

    def loss(self, a, y):

        def square(a):
            return a*a

        loss = 0

        for y_pred, y_actual in zip(a, y):
            x = np.sum(0.5*square(y_pred-y_actual))
            loss += x

        return loss/len(a)

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
