"""
Generalised Linear Model

References:

ToDo:
Reguralisation
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
import scipy.optimize

from bareml import BinaryClassifier, Regressor
from bareml.utils.manipulators import prob2binary
from bareml.utils.model_tuning import initialise_random
from bareml.utils.probability_distribution import Bernoulli, Binomial, Poisson, Gaussian


class _GLM:
    """ Generalised Linear Model """
    
    def __init__(self, prob):
        self.prob = prob
        self.params = None

    def fit(self, X, y):

        # add a column of 1 for bias 
        X = np.insert(X, 0, 1, axis=1)

        # init params
        params = initialise_random(X.shape[1])

        minimise_func = lambda params, X, y: -np.sum(self.prob.llh(y, self.prob.link(params @ X.T), opt_for_minimise=True))

        opt_params = scipy.optimize.minimize(minimise_func, params, args=(X, y))

        self.params = opt_params

        return self

    def predict(self, X):
        # add a column of 1 for bias 
        X = np.insert(X, 0, 1, axis=1)

        return self.prob.link(self.params.x @ X.T)


class LogisticRegression(_GLM, BinaryClassifier):
    """ 
    Logistic Regression 
    where the target variable is {0,1}
    """
    def __init__(self):
        super().__init__(Bernoulli())

    def fit(self, X, y):
        X, y = self._validate_Xy(X,y)
        return super().fit(X, y)

    def predict(self, X):
        X = self._validate_X(X)
        return prob2binary(super().predict(X))

    def predict_proba(self, X):
        X = self._validate_X(X)
        return super().predict(X)


class LogisticRegressionBinom(_GLM, BinaryClassifier):
    """ 
    Logistic Regression 
    where the target variable is described as a pair of values (n, k)
    """
    def __init__(self):
        super().__init__(Binomial())

    def fit(self, X, y):
        X, y = self._validate_Xy(X,y)
        return super().fit(X, y)

    def predict(self, X):
        X = self._validate_X(X)
        return prob2binary(super().predict(X))

    def predict_proba(self, X):
        X = self._validate_X(X)
        return super().predict(X)


class PoissonRegression(_GLM, Regressor):
    """ Poisson Regression """
    def __init__(self):
        super().__init__(Poisson())

    def fit(self, X, y):
        X, y = self._validate_Xy(X,y)
        return super().fit(X, y)

    def predict(self, X):
        X = self._validate_X(X)
        return super().predict(X)


class LinearRegression(_GLM, Regressor):
    """ Linear Regression """
    def __init__(self):
        super().__init__(Gaussian())

    def fit(self, X, y):
        X, y = self._validate_Xy(X,y)
        return super().fit(X, y)

    def predict(self, X):
        X = self._validate_X(X)
        return super().predict(X)