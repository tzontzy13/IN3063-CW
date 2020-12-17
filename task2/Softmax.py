import numpy as np

class Softmax():
    def __init__(self):
        pass

    # formula can be found in lecture
    def softmax(self, x):

        e = np.exp(x - np.max(x))
        output = e / np.sum(e)
        return output

    # apply relu to numpy.array
    def forward(self, zs):
        output = self.softmax(zs)
        return output