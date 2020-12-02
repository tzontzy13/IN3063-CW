import random
import numpy as np


class MLP():

    # initialize weights and biases randomly
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = []
        self.weights = []

        self.num_layers = len(self.sizes)

        # self.biases2 = []
        # self.weights2 = []

        # for i in range(1, len(sizes)):
        #     self.biases2.append(np.random.randn(sizes[i]))

        # for i, j in zip(sizes[:-1], sizes[1:]):
        #     self.weights2.append(np.random.randn(j, i))

        # self.biases2 = np.array(self.biases2, dtype=object)
        # self.weights2 = np.array(self.weights2, dtype=object)

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # deprecated
    def forward(self, xs):

        all_layers = []
        output_layer = []
        # hidden_layer = []
        num_of_layers = len(self.sizes) - 1

        for x in xs:
            image_layer_activations = []

            inputs = x
            for i in range(num_of_layers):
                single_layer_activations = []

                for b, w in zip(self.biases[i], self.weights[i]):
                    single_layer_activations.append(
                        self.sigmoid(np.dot(w, inputs)+b))

                image_layer_activations.append(single_layer_activations)
                inputs = single_layer_activations

            all_layers.append(image_layer_activations)

        for x in all_layers:
            output_layer.append(x[-1])

        # for x in all_layers:
        #     hidden_layer.append(x[0])

        return np.array(output_layer, dtype=object)
        # , np.array(hidden_layer, dtype=object), np.array(all_layers, dtype=object)

    def forward2(self, x):

        # data structures for activations and Z's
        # activations = sigmoid(sum of weight * prev_act + bias)
        # Z = sum of weight * prev_act + bias
        # helps with the derivative of activations depending on Z's (sigmoid_derivated)
        image_layer_activations = [x]
        image_layer_zs = []

        inputs = x
        num_of_layers = len(self.sizes) - 1

        # for each layer of the NN
        for i in range(num_of_layers):

            # data strucutres for activations and Z's of that layer
            single_layer_activations = []
            single_layer_zs = []

            # for each weight and bias in that layer
            # bias shape is for each neuron, one value
            # weights shape is for each neuron,
            # we have a weight for each neuron in the prev layer
            for b, w in zip(self.biases[i], self.weights[i]):
                single_layer_activations.append(
                    self.sigmoid(np.dot(w, inputs)+b))
                single_layer_zs.append(np.dot(w, inputs)+b)

            # append each layer of activ and Z's in the all activ and Z's data structure
            image_layer_activations.append(
                np.array(single_layer_activations, dtype=float))
            image_layer_zs.append(np.array(single_layer_zs, dtype=float))
            # inputs becomes the activations of the last layer,
            # so i have the right number of neurons for the next iteration
            inputs = single_layer_activations

        # return all activ and Z's, for all layers
        return np.array(image_layer_activations, dtype=object), np.array(image_layer_zs, dtype=object)

    # calculation of MSE - error used to get the cost
    def loss(self, a, y):

        def square(a):
            return a*a

        loss = 0

        # sum of squared differences
        # if outputul is [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.8, 0.1, 0.1]
        # and Y_PRED is  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        # cost is direference of each collumn, squared, sum together
        for y_pred, y_actual in zip(a, y):
            x = np.sum(0.5*square(np.subtract(y_pred, y_actual)))
            loss += x

        return loss/len(a)

    def evaluate(self, test_data, test_targets):
        def feedforward(a):
            for b, w in zip(self.biases, self.weights):
                a = self.sigmoid(np.dot(w, a)+b)
            return a

        test_results = [(np.argmax(feedforward(x)), np.argmax(y))
                        for (x, y) in zip(test_data, test_targets)]
        pred_sum = 0
        for (x, y) in test_results:
            pred_sum += int(x == y)

        return sum(int(x == y) for (x, y) in test_results)/len(test_data)*100

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def relu_derivated(x):
        if x > 0:
            return 1
        else:
            return 0


def sigmoid_derivated(n):
    return MLP.sigmoid(n)*(1-MLP.sigmoid(n))
