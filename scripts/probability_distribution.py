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

from scripts.activation_functions import Sigmoid

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
    """ Bernoulli distribution """

    def pmf(self, p, y):
        """
        Probability mass function.

        Parameters
        ----------
        p: float [0,1]
            Parameter of Bernoulli distribution
        y: int {0,1}
            Realised value of the random variable

        Returns
        -------
        float
            Probability
        """
        return ( p ** y ) * ( (1-p) ** (1-y) )
            
    def stats(self, p):
        """
        Calculates statistics

        Parameters
        ----------
        p: float [0,1]
            Parameter of Bernoulli distribution

        Returns
        -------
        mu: float
            Mean
        var: float
            Variance
        """
        mu = p
        var = p * (1-p)
        return mu, var

    def link(self, z):
        """ Link function (for GLM) """
        return 1 / (1 + np.exp(-z))
    
    def llh(self, p, y):
        """
        Calculates log likelihood.

        Parameters
        ----------
        p: np.array (n,)
            parameter for each data point
            each element is in [0,1]
            n: number of data points

        y: np.array (n,)
            observed value of each data point
            each element is either 0 or 1
            n: number of data points

        Returns
        -------
        float
            Log likelihood
        """
        # Avoid log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.sum(y * np.log(p) + (1-y) * np.log(1-p))


class Binomial(DiscreteProbabilityDistribution):
    """ Binomial distribution """

    def pmf(self, p, n, k):
        """
        Probability mass function.

        Parameters
        ----------
        p: float [0,1]
            Parameter of Binomial distribution
        n: int 0 < n
            number of trial
        k: int 0 <= k <= n
            number of success

        Returns
        -------
        float
            Probability
        """
        return ncr(n,k) * ( p ** k ) * ( (1-p) ** (n-k) )

    def stats(self, p, n):
        """
        Calculates statistics

        Parameters
        ----------
        p: float [0,1]
            Parameter of Binomial distribution
        n: int 0 < n
            Number of trial

        Returns
        -------
        mu: float
            Mean
        var: float
            Variance
        """
        mu = n * p
        var = n * p * (1-p)
        return mu, var

    def link(self, z):
        """ Link function (for GLM) """
        return 1 / (1 + np.exp(-z))

    def llh(self, p, y):
        """
        Calculates ~ log likelihood.
        Note that this function doesn't return exact log likelihood, 
        but returns a value ~ log likelihood.
        Actual log likelihood of binomial distribution is
        \sum_i{ log(n_iCk_i) + k_i log(p_i) + (n_i - k_i) log(1 - p_i) }
        But this function returns \sum_i{ k_i log(p_i) + (n_i - k_i) log(1 - p_i) }
        and it's totally fine for argmax calculation purpose. 

        Parameters
        ----------
        p: np.array (m,)
            parameter for each data point
            each element is in [0,1]
            m: number of data points

        y: np.array (m,2)
            each row is a tuple of (n, k)
            where m is the number of trial, and
            k is the number of success in observed data point.
            m: number of data points

        Returns
        -------
        float
            ~ log likelihood
        """
        # Avoid log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        n = y[:,0]
        k = y[:,1]
        return np.sum(k * np.log(p) + (n-k) * np.log(1-p))


class Poisson(DiscreteProbabilityDistribution):
    """ Poisson distribution """

    def pmf(self, p, k):
        """
        Probability mass function.

        Parameters
        ----------
        p: float p > 0
            Parameter of Poisson distribution (usually called lambda)
        k: int 0 in {0, 1, 2, ...}
            Realised value of the random variable

        Returns
        -------
        float
            Probability
        """
        return ( p ** k ) * np.exp(-p) / math.factorial(k)

    def stats(self, p):
        """
        Calculates statistics

        Parameters
        ----------
        p: float p > 0
            Parameter of Poisson distribution (usually called lambda)

        Returns
        -------
        mu: float
            Mean
        var: float
            Variance
        """
        mu = var = p
        return mu, var

    def link(self, z):
        """ Link function (for GLM) """
        return np.exp(z)

    def llh(self, p, y):
        """
        Calculates ~ log likelihood.
        Note that this function doesn't return exact log likelihood, 
        but returns a value ~ log likelihood.
        Actual log likelihood of poisson distribution is
        \sum_i{ y_i log(p_i) - p_i - log(y_i!) }
        But this function returns \sum_i{ y_i log(p_i) - p_i  }
        and it's totally fine for argmax calculation purpose. 

        Parameters
        ----------
        p: np.array (n,)
            parameter for each data point
            each element is a float > 0
            n: number of data points

        y: np.array (n,)
            observed value of each data point
            each element is in {0,1,2,...}
            n: number of data points

        Returns
        -------
        float
            ~ log likelihood
        """
        return np.sum(y * np.log(p) - p)