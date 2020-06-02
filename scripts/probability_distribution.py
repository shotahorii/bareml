"""
Probability Distributions

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import operator as op
from functools import reduce
from abc import ABC, abstractmethod
import math
import numpy as np

def ncr(n, r):
    """
    Calculates nCr in efficient manner. 
    This function is not my original code, but copied from below url.
    https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    """
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


class DiscreteProbabilityDistribution(ABC):

    @abstractmethod
    def pmf(self):
        pass
    
    @abstractmethod
    def stats(self):
        pass


class Bernoulli(DiscreteProbabilityDistribution):

    def __init__(self, p):
        if p > 1 or p < 0:
            raise ValueError("invalid input.")

        self.p = p

    def pmf(self, x):

        if x != 0 and x != 1:
            raise ValueError("input needs to be either 0 or 1.")

        return (self.p ** x) * ( (1-self.p) ** (1-x) )
            
    def stats(self):
        mu = self.p
        var = self.p * (1-self.p)
        return mu, var

class Binomial(DiscreteProbabilityDistribution):

    def __init__(self, p, n):
        if p > 1 or p < 0 or n < 1 or round(n) != n:
            raise ValueError("invalid input.")

        self.p = p
        self.n = n

    def pmf(self, k):

        if k > self.n or round(k) != k:
            raise ValueError("invalid input.")

        return ncr(self.n,k) * (self.p ** k) * ( (1-self.p) ** (self.n-k) )

    def stats(self):
        mu = self.n * self.p
        var = self.n * self.p * (1-self.p)
        return mu, var
