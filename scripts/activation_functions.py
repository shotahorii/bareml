"""
Activation functions.

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import numpy as np

class Sigmoid:

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        return self.value(x) * (1 - self.value(x))