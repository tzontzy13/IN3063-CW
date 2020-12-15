import numpy as np


class Dense():

    def __init__(self, neurons_in_input, neurons_in_output, l_r=0.05):

        self.neurons_in_input = neurons_in_input
        self.neurons_in_output = neurons_in_input

        self.l_r = l_r

        # docs.scipy. 2016.
        #  numpy.random.normal. 
        # [ONLINE] Available at: https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.normal.html. 
        # [Accessed 15 December 2020].

        # we used the above website to change the SCALE ATTRIBUTE from 1.0 to 0.1 for the np.random.normal function
        # -> we got the below error when scale = 1
        # RuntimeWarning: overflow encountered in exp
        # softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        # END OF ERROR
        # fixed error with scale = 0.1, because now the numbers are smaller and there is no overflow in RELU + RELU layers, 
        # at the softmax function in the gradient of the loss
        self.weights = np.random.normal(scale=1,
                                        size=(neurons_in_input, neurons_in_output))
        self.biases = np.random.normal(size=(neurons_in_output))

    # Returns activations for dense layer
    def forward(self, activations):

        # this is (weights * activations) + bias
        # formula in lecture and lab 7
        return np.dot(activations, self.weights) + self.biases

    # returns the sigma_z for dense layer
    # also updates the weights and biases using the gradients for weights and biases and LEARNING RATE
    # a similar formula was used in lecture and lab 7
    def backward(self, activation, next_layer_sigma_z):

        # sigma_z for this layer
        # formula in lecture and lab 7
        # for each image, cost at each neuron
        # derivative of loss/cost (next_layer_sigma_z) with respect to activation is next_layer_sigma_z * weights
        # because (w1a1 + w2a2 + ... + wnan) + b derivated with respect to activations (a1, a2, ..., an) is just the weights
        # you transpose the weights because you need the shape of the next_layer_sigma_z
        # for example, if weights is shape (200,100) and next_layer_sigma_z is shape (minibatch_size, 100)
        # you need your new sigma_z to be of shape (minibatch_size, 200), so you can use it in the next iteration of backward pass
        sigma_z = np.dot(next_layer_sigma_z, self.weights.transpose())

        # gradient of the weights of this layer
        # formula in lecture and lab 7
        # derivative of loss/cost (next_layer_sigma_z) with respect to weights is next_layer_sigma_z * activations
        # must be of shape (200, 100) (same as weights, so we can update)
        # lets say activation is of shape (minibatch_size, 200) (for each image, 200 neurons)
        # and next_layer_sigma_z (next layer error) is (minibatch_size, 100) (for each image, 100 neurons)
        # we transpose activation in (200, minibatch size)
        # we can use dot for shapes (200, minibatch size) , (minibatch size, 100)
        # and we get shape for gradient of weights (200,100)
        gradient_w = np.dot(activation.transpose(), next_layer_sigma_z)

        # gradient of the biases of his layer
        # formula in lecture and lab 7
        # derivative of loss/cost (next_layer_sigma_z) with respect to biases is next_layer_sigma_z
        # since bias is of shape (100), gradient must be of shape (100)
        # shape of next_layer_sigma_z is (32, 100) for each image, 100 neurons
        # so we need to average next_layer_sigma_z so that the gradient for the bias is shape (100)
        gradient_b = np.zeros(shape=(len(next_layer_sigma_z[0])))

        # add up all gradients for each image in minibatch
        for single_image_sigma_z in next_layer_sigma_z:
            gradient_b += single_image_sigma_z

        # divide by the length of the minibatch
        gradient_b = gradient_b / len(next_layer_sigma_z[0])

        # update weights and biases with gradient
        # formula in lecture and lab 7
        self.weights -= self.l_r * gradient_w
        self.biases -= self.l_r * gradient_b

        return sigma_z
