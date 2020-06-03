"""
Generalised Linear Model

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 124-127.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 511-512.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 463-481.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 54-69.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
import scipy.optimize
from scripts.probability_distribution import Bernoulli, Binomial, Poisson

class GLM:
    
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
    pass

class LogisticRegressionBinom(GLM):
    pass

class PoissonRegresssion(GLM):
    pass