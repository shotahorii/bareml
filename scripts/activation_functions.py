"""
Activation functions.

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import numpy as np

class Sigmoid:

    @classmethod
    def value(self, x):
        return 1 / (1 + np.exp(-x))
    
    @classmethod
    def derivative(self, x):
        return self.value(x) * (1 - self.value(x))