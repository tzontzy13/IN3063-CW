import numpy as np

class Sigmoid():
    def __init__(self):
        pass

    # formula for sigmoid, can be found in lectures
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # apply sigmoid to a numpy.array
    def forward(self, zs):
        output = self.sigmoid(zs)
        return output
    
    # derivative of sigmoid, used for calculating cost
    # derivative of the cost/error with respect to sigmoid is: cost * sigmoid derivated
    # chain rule
    # formula can be found in the lectures
    def backward(self, zs, next_layer_sigma_z):
        sigmoid_derivated = self.sigmoid(zs) * (1 - self.sigmoid(zs))
        output = next_layer_sigma_z * sigmoid_derivated
        return output