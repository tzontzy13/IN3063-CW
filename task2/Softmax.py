import numpy as np

class Softmax():
    def __init__(self):
        pass

    # done
    # formula can be found in lecture
    def softmax(self, x):

        e = np.exp(x - np.max(x))
        output = e / np.sum(e)
        return output

    # done
    # apply relu to numpy.array
    def forward(self, zs):
        output = self.softmax(zs)
        return output
    
    # done
    # copiat
    def backward(self, zs, next_layer_sigma_z):
        # calculate derivative of next_layer_sigma_z (error) with respect to zs
        relu_derivated = zs > 0
        output = next_layer_sigma_z * relu_derivated
        return output