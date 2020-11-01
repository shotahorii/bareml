"""
Kernel Regression

Author: Shota Horii <sh.sinker@gmail.com>
Test: tests/kernel_regression.py

References:
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 494-495.
"""


import math
import numpy as np

from ..base import Regressor
from ..utils.manipulators import StandardScaler
from ..utils.kernels import linear_kernel, polynomial_kernel, sigmoid_kernel, rbf_kernel


class KernelRidgeRegression(Regressor):
    """
    Kernel Ridge Regression

    Parameters
    ----------
    alpha: float >=0
        Regularisation strength
    
    kernel: string {'linear', 'rbf', 'polynomial', 'sigmoid'}
        Kernel function
    
    gamma: float (-inf, inf)
        parameter of rbf, polynomial and sigmoid kernel. 
        ignored when kernel is other kernels.

    degree: int (-inf, inf)
        parameter of polynomial kernel. 
        ignored when kernel is other kernels.
    
    coef0: float (-inf, inf)
        parameter of polynomial and sigmoid kernel. 
        ignored when kernel is other kernels.
    """

    def __init__(self, alpha=0, kernel='linear', gamma=None, degree=3, coef0=1):
        self.alpha = alpha

        self.X = None
        self.a = None

        if kernel == 'linear':
            self.kernel = linear_kernel
            self.params = {}
        elif kernel == 'rbf':
            self.kernel = rbf_kernel
            self.params = {'gamma':gamma}
        elif kernel == 'polynomial':
            self.kernel = polynomial_kernel
            self.params = {'degree':degree, 'gamma':gamma, 'coef0':coef0}
        elif kernel == 'sigmoid':
            self.kernel = sigmoid_kernel
            self.params = {'gamma':gamma, 'coef0':coef0}
        else:
            raise ValueError('Invalid Kernel.')

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray (n, d) 
            n: number of samples
            d: number of features
        
        y: np.ndarray (n,)
            n: number of samples

        Returns
        -------
        self: KernelRidgeRegression
        """

        # store the input X as we use this for prediction
        self.X = X

        # K will be a (n, n) matrix
        n = X.shape[0]
        K = self.kernel(X, **self.params)

        # self.a will be a (n,) vector
        self.a = np.linalg.pinv(K + self.alpha * np.eye(n)) @ y

        return self

    def _predict(self, X):
        """
        Parameters
        ----------
        X: np.ndarray (m, d) 
            m: number of samples to predict
            d: number of features

        Returns
        -------
        y_pred: np.ndarray (m,)
            m: number of samples to predict
        """
        
        n = self.X.shape[0] # number of trained samples 
        m = X.shape[0] # number of samples to predict

        # K will be a (m, n) matrix
        # why don't we make it as (n, m)?
        # -> Considering the meaning of the matrix, 
        # we want a row to be representing a sample x' to predict.
        # each row is a (n,) vector, which is representing k(x', x_i)
        # where x_i (i=1~n) is each of trained samples.
        K = self.kernel(X, self.X, **self.params)

        # as self.a is a (n,) vector, 
        # (m, n) @ (n,) -> (m,)
        # hence y_pred will be a (m,) vector
        y_pred = K @ self.a

        return y_pred