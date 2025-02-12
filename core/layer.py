"""
This module defines the behavior of a neural network linear layer

"""
from mpmath import mp


class Layer:
    pass


class LinearLayer(Layer):
    def __init__(self, weight: mp.matrix, bias: mp.matrix):
        super().__init__()

        self.weight = weight
        self.bias = bias
