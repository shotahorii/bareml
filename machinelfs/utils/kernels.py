"""
Kernel functions

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from machinelfs.utils.distances import l2_norm

class LinearKernel:

    def __call__(self, x1, x2):
        return x1 @ x2

class PolynomialKernel:

    def __init__(self, p, c):
        self.p = p
        self.c = c

    def __call__(self, x1, x2):
        return (x1 @ x2 + self.c) ** self.p

class RBFKernel:

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, x1, x2):
        return np.exp(-self.gamma * (l2_norm(x1 - x2) ** 2))

class SigmoidKernel:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x1, x2):
        return np.tanh(self.a * (x1 @ x2) + self.b)