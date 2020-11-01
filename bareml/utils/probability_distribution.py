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

from .misc import ncr
from .activation_functions import Sigmoid


class DiscreteProbabilityDistribution(ABC):

    @abstractmethod
    def pmf(self):
        pass
    
    @abstractmethod
    def stats(self):
        pass

class ContinuousProbabilityDistribution(ABC):

    @abstractmethod
    def pdf(self):
        pass
    
    @abstractmethod
    def stats(self):
        pass


class Bernoulli(DiscreteProbabilityDistribution):
    """ Bernoulli distribution """

    def __init__(self):
        self.param_names = ['p']

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
    
    def llh(self, y, p, opt_for_minimise=False):
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

        opt_for_minimise: bool
            if true, optimise for minimisation
            (Doesn't do any for Bernoulli)

        Returns
        -------
        llh: np.array (n,)
            Log likelihood
            n: number of data points
        """
        # Avoid log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return y * np.log(p) + (1-y) * np.log(1-p)

    def mle(self, y, param='p'):
        """
        Estimates parameter from data by Maximum Likelihood Estimation.

        Parameters
        ----------
        y: np.ndarray (n,) int {0,1}
            data points which we assume realisations from a Bernoulli distribution
        
        param: always 'p'
            parameter to estimate.
            no need to specify as only 1 param in Bern. distribution. 
            still taken as an input for consistency purpose.

        Returns
        -------
        float
            estimated value of the parameter p
        """

        if param=='p':
            return y.mean()
        else:
            raise ValueError('param must be "p".')


class Binomial(DiscreteProbabilityDistribution):
    """ Binomial distribution """
    def __init__(self):
        self.param_names = ['p']

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

    def llh(self, y, p, opt_for_minimise=False):
        """
        Calculates log likelihood.
        \sum_i{ log(n_iCk_i) + k_i log(p_i) + (n_i - k_i) log(1 - p_i) }
        if opt_for_minimise is true, calcluates ~log likelihood as below
        \sum_i{ k_i log(p_i) + (n_i - k_i) log(1 - p_i) }

        Parameters
        ----------
        p: np.array (m,)
            parameter for each data point
            each element is in [0,1]
            m: number of data points

        y: np.array (m,2)
            each row is a tuple of (n, k)
            where n is the number of trial, and
            k is the number of success in observed data point.
            m: number of data points

        opt_for_minimise: bool
            if true, optimise for minimisation

        Returns
        -------
        float
            log likelihood
        """
        # Avoid log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)

        n = y[:,0]
        k = y[:,1]

        if opt_for_minimise:
            return k * np.log(p) + (n-k) * np.log(1-p)
        else:
            nck = np.array([ncr(nk[0], nk[1]) for nk in y])
            return np.log(nck) + k * np.log(p) + (n-k) * np.log(1-p)

class Poisson(DiscreteProbabilityDistribution):
    """ Poisson distribution """
    def __init__(self):
        self.param_names = ['lmd']

    def pmf(self, lmd, k):
        """
        Probability mass function.

        Parameters
        ----------
        lmd: float lmd > 0
            Parameter of Poisson distribution (usually called lambda)
        k: int 0 in {0, 1, 2, ...}
            Realised value of the random variable

        Returns
        -------
        float
            Probability
        """
        return ( lmd ** k ) * np.exp(-lmd) / math.factorial(k)

    def stats(self, lmd):
        """
        Calculates statistics

        Parameters
        ----------
        lmd: float lmd > 0
            Parameter of Poisson distribution (usually called lambda)

        Returns
        -------
        mu: float
            Mean
        var: float
            Variance
        """
        mu = var = lmd
        return mu, var

    def link(self, z):
        """ Link function (for GLM) """
        return np.exp(z)

    def llh(self, y, lmd, opt_for_minimise=False):
        """
        Calculates log likelihood.
        \sum_i{ y_i log(p_i) - p_i - log(y_i!) }
        if opt_for_minimise is true, calcluates ~log likelihood as below
        \sum_i{ y_i log(p_i) - p_i  }

        Parameters
        ----------
        lmd: np.array (n,)
            parameter for each data point
            each element is a float > 0
            n: number of data points

        y: np.array (n,)
            observed value of each data point
            each element is in {0,1,2,...}
            n: number of data points

        opt_for_minimise: bool
            if true, optimise for minimisation

        Returns
        -------
        float
            log likelihood
        """

        if opt_for_minimise:
            return y * np.log(lmd) - lmd
        else:
            fct = np.array([math.factorial(v) for v in y])
            return y * np.log(lmd) - lmd - np.log(fct)


class Gaussian(ContinuousProbabilityDistribution):
    """ Gaussian distribution (univariate) """

    def __init__(self):
        self.param_names = ['mu','var']

    def pdf(self, mu, var, x):
        """
        Probability mass function.

        Parameters
        ----------
        mu:
        var:
        x:

        Returns
        -------
        float
            Probability density
        """
        normalising_constant = 1/np.sqrt(2 * math.pi * var)
        kernel = np.exp( -np.power(x-mu, 2) / (2*var) )
        return normalising_constant * kernel

    def stats(self, mu, var):
        """ For Gaussian, parameters are already mean and variance... """
        return mu, var

    def link(self, z):
        """ Link function (for GLM) = identity function """
        return z

    def llh(self, y, mu, var=None, opt_for_minimise=False):
        """
        Calculates log likelihood.
        \sum_i{ -1/2 log(2*pi*q_i) - 1/(2*q_i) (y_i - p_i)^2 }
        if opt_for_minimise is true, calcluates ~log likelihood as below
        note that this is for an optimisation for mu not var
        \sum_i{ -(y_i - p_i)^2 }

        Parameters
        ----------
        mu: np.array (n,) float in (-inf, inf)
            'mean' parameter for each data point
            n: number of data points

        var: np.array (n,) float in (0, inf)
            'variance' parameter for each data point
            n: number of data points

        y: np.array (n,) float in (-inf, inf)
            observed value of each data point
            n: number of data points

        opt_for_minimise: bool
            if true, optimise for minimisation

        Returns
        -------
        float
            log likelihood
        """
        if opt_for_minimise:
            return -np.power(y-mu, 2)
        else:
            return -np.log(2*math.pi*var)/2 - np.power(y-mu, 2)/(2*var)

    def mle(self, y, param):
        """
        Estimates parameters from data by Maximum Likelihood Estimation.

        Parameters
        ----------
        y: np.ndarray (n,) float in (-inf, inf)
            data points which we assume realisations from a Gaussian distribution
        
        param: string {'mu', 'var'}
            parameter to estimate

        Returns
        -------
        float
            estimated value of the parameter (mu or var)
        """

        if param=='mu':
            return y.mean()
        elif param=='var':
            return y.var()
        else:
            raise ValueError('param must be either "mu" or "var".')
