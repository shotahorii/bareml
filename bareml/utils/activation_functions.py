"""
Activation functions.

Author: Shota Horii <sh.sinker@gmail.com>

References:

"""


import numpy as np

class Identity:

    def __call__(self, x):
        return x
    
    def gradient(self, x):
        return 1.0


class Sigmoid:

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Softmax:

    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])

        # Avoid overflow
        x = x - np.array([x.max(axis=1)]).T

        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x,axis=1)[np.newaxis].T
        return exp_x / sum_exp_x

    def gradient(self, x):
        pass