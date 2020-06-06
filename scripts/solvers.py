"""
Solvers

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

from abc import ABC, abstractmethod
import math
import numpy as np

class Solver(ABC):

    @abstractmethod
    def solve(self):
        pass

class PInv(Solver):

    def __init__(self, alpha=0):
        self.alpha = alpha

    def solve(self, X, y):
        d = X.shape[1]
        return np.dot( np.dot(np.linalg.pinv( np.dot(X.T, X) + self.alpha * np.eye(d) ), X.T), y)

class LassoISTA(Solver):
    
    def __init__(self, alpha=0):
        self.alpha = alpha

    def solve(self, X, y):
        pass