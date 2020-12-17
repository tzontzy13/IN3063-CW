# Referece: 1
# Aayush Agrawal. 2020
# Building Neural Network from scratch | by Aayush Agrawal | Towards Data Science.
# [ONLINE] Available at: https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
# [Accessed 15 December 2020]

# Referece: 2
# Raul Gomez Bruballa. 2018
# Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names.
# [ONLINE] Available at: https://gombru.github.io/2018/05/23/cross_entropy_loss/.
# [Accessed 15 December 2020].

import numpy as np

from Softmax import Softmax

class CrossEntropy():

    def __init__(self):
        # never used
        # we are using a custom_softmax, that is referenced below
        self.softmax = Softmax()
        pass

    # Reference 2 - formula from images and text
    # scroll down to Categorical Cross-Entropy loss ( Negative Likelihood Loss )
    # we built this function with the help of the first 2 images and the text associated with them
    def softmax_cross_entropy_loss(self, zs, targets):

        # building the predictions list
        # a list of what was predicted
        # get targets and zs
        # target is an int
        # zs are an array of shape (10,1)
        # zs[target] is what is predicted
        predictions = []

        for output_vector, answer in zip(zs, targets):
            predictions.append(output_vector[answer])

        # custom softmax
        # get zs and predicitons as input
        # computes the exponential of predictions
        # and the sum of the exponential of the zs
        # just like in the second picture (NOT the first) from the
        # referenced wesite, under Categorical Cross-Entropy loss
        def custom_softmax(zs, predictions):
            e = np.exp(predictions)
            # axis = 1 means sum over rows
            s = np.sum(np.exp(zs), axis=1)
            return e / s

        # get softmax value from above function
        custom_softmax_value = custom_softmax(zs, predictions)

        # the Categorical Cross-Entropy (NLL) loss is calculated using -log of the softmax
        # just like in the second picture (NOT the first) from the
        # referenced wesite, under Categorical Cross-Entropy loss
        cost = -np.log(custom_softmax_value)

        # returns the Categorical Cross-Entropy loss value
        return cost

    # Reference 1 - code
    # this is the website where we got our "skeleton" from
    # we copied this whole function which computes the gradient of the cross entropy with logits
    # gradient of cross entropy
    def grad_softmax_crossentropy_with_logits(self, logits, reference_answers):

        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        return (- ones_for_answers + softmax) / logits.shape[0]
