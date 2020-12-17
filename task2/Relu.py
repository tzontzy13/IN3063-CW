import numpy as np

class Relu():
    def __init__(self):
        pass

    # formula can be found in lecture
    def relu(self, x):
        output = np.maximum(0.0, x)
        return output

    # apply relu to numpy.array
    def forward(self, zs):
        output = self.relu(zs)
        return output
    
    # Reference: Github - Yusuke Sugomori Repositories. 2020
    # Title: Dropout Neural Networks (with ReLU).
    # [ONLINE] Available at: https://gist.github.com/yusugomori/cf7bce19b8e16d57488a.
    # [Accessed 15 December 2020].
    # copied line 24 - relu_derivated = zs > 0
    def backward(self, zs, next_layer_sigma_z):
        # Calculates derivative of next_layer_sigma_z (error) with respect to zs
        relu_derivated = zs > 0

        output = next_layer_sigma_z * relu_derivated

        return output
