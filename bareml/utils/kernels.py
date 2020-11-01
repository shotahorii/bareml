"""
Kernel functions

Author: Shota Horii <sh.sinker@gmail.com>

References:
https://scikit-learn.org/stable/modules/metrics.html#pairwise-metrics-affinities-and-kernels

"""

import math
import numpy as np

from .distances import l2_norm


def kernel_eligible_pair(X, Y):
    """
    Validate X and Y if those are eligible to compute karnel
    
    Parameters
    ----------
    X: np.ndarray (n, d) of real (-inf, inf)
        n: number of samples in X
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    Y: np.ndarray (m, d) of real (-inf, inf)
        m: number of samples in Y
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    Retures
    -------
    K: np.ndarray (n, m)
        Kernel matrix
    """
    def _eligible(A):
        # make sure it's a numpy array
        A = np.array(A)

        # make sure data type and data shape is good 
        if A.dtype not in ['int64','float64','uint8']:
            raise ValueError('Data type of input needs to be int or float.')
        elif np.isnan(A).any():
            raise ValueError('There is at least 1 null element in the input.')
        elif np.isinf(A).any():
            raise ValueError('There is at least 1 inf/-inf element in the input.')
        elif A.ndim > 2:
            raise ValueError('input cannot be 3d or larger dimension.')

        # make sure input is 2d array shape
        if A.ndim == 1:
            A = np.array([A])
        elif A.ndim == 0:
            A = np.array([[A]])

        return A
    
    # if Y is not given, compute kernel between X and X
    if Y is None:
        Y = X

    X, Y = _eligible(X), _eligible(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError('Number of features in X and number of features in Y should be same.')

    return X, Y


def linear_kernel(X, Y=None):
    """
    Computes linear kernel between X and Y
    
    Parameters
    ----------
    X: np.ndarray (n, d) of real (-inf, inf)
        n: number of samples in X
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    Y: np.ndarray (m, d) of real (-inf, inf)
        m: number of samples in Y
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    Retures
    -------
    K: np.ndarray (n, m)
        Kernel matrix
    """
    X, Y = kernel_eligible_pair(X, Y)
    return X @ Y.T


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    Computes polynomial kernel between X and Y
    
    Parameters
    ----------
    X: np.ndarray (n, d) of real (-inf, inf)
        n: number of samples in X
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    Y: np.ndarray (m, d) of real (-inf, inf)
        m: number of samples in Y
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    degree: int (-inf, inf)
    gamma: float (-inf, inf)
    coef0: float (-inf, inf)

    Retures
    -------
    K: np.ndarray (n, m)
        Kernel matrix
    """

    X, Y = kernel_eligible_pair(X, Y)
    
    if gamma is None:
        # defaults to 1.0 / n_features
        gamma = 1.0 / len(X[1])

    return (gamma * (X @ Y.T) + coef0) ** degree


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """
    Computes sigmoid kernel between X and Y
    
    Parameters
    ----------
    X: np.ndarray (n, d) of real (-inf, inf)
        n: number of samples in X
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    Y: np.ndarray (m, d) of real (-inf, inf)
        m: number of samples in Y
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    gamma: float (-inf, inf)
    coef0: float (-inf, inf)

    Retures
    -------
    K: np.ndarray (n, m)
        Kernel matrix
    """

    X, Y = kernel_eligible_pair(X, Y)
    
    if gamma is None:
        # defaults to 1.0 / n_features
        gamma = 1.0 / len(X[1])

    return np.tanh(gamma * (X @ Y.T) + coef0)


def rbf_kernel(X, Y=None, gamma=None):
    """
    Computes RBF kernel between X and Y
    
    Parameters
    ----------
    X: np.ndarray (n, d) of real (-inf, inf)
        n: number of samples in X
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    Y: np.ndarray (m, d) of real (-inf, inf)
        m: number of samples in Y
        d: number of features
        if a 1d array (d,) is given, it's automatically 
        converted into 2d array (1,d)
        if a scalar is given, it's automatically 
        convered into 2d array (1,1) where d=1

    gamma: float (-inf, inf)

    Retures
    -------
    K: np.ndarray (n, m)
        Kernel matrix
    """

    X, Y = kernel_eligible_pair(X, Y)
    
    if gamma is None:
        # defaults to 1.0 / n_features
        gamma = 1.0 / len(X[1])

    n, m = len(X), len(Y)
    K = -gamma * np.array([[ l2_norm(X[i] - Y[j]) for j in range(m)] for i in range(n)]) ** 2
    return np.exp(K)