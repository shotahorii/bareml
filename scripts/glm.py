"""
Generalised Linear Model

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
import scipy.optimize
from scripts.probability_distribution import Bernoulli, Binomial, Poisson

class GLM:
    """ Generalised Linear Model """
    
    def __init__(self, prob):
        self.prob = prob
        self.params = None

    def _initialise_params(self, n_params):
        """
        Initialise the value of parameters to estimate

        Parameters
        ----------
        n_params: int
            number of parameters to estimate
        """

        # initialise params in [-1/sqrt(n), 1/sqrt(n)) 
        # why? see below articles. 
        # https://leimao.github.io/blog/Weights-Initialization/
        # https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
        limit = 1 / math.sqrt(n_params)
        return np.random.uniform(-limit, limit, n_params)

    def fit(self, X, y):

        # add a column of 1 for bias 
        X = np.insert(X, 0, 1, axis=1)

        # init params
        params = self._initialise_params(X.shape[1])

        minimise_func = lambda params, X, y: -self.prob.llh(self.prob.link(params @ X.T), y)

        opt_params = scipy.optimize.minimize(minimise_func, params, args=(X, y))

        self.params = opt_params

    def predict(self, X):
        # add a column of 1 for bias 
        X = np.insert(X, 0, 1, axis=1)

        return self.prob.link(self.params.x @ X.T)


class LogisticRegression(GLM):
    """ 
    Logistic Regression 
    where the target variable is {0,1}
    """
    def __init__(self):
        super().__init__(Bernoulli())


class LogisticRegressionBinom(GLM):
    """ 
    Logistic Regression 
    where the target variable is described as a pair of values (n, k)
    """
    def __init__(self):
        super().__init__(Binomial())

class PoissonRegresssion(GLM):
    """ Poisson Regression """
    def __init__(self):
        super().__init__(Poisson())