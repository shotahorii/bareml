"""
Model Tuning

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from mlfs.utils.misc import supremum_eigen

#########################
# Weight Initialisation #
#########################

def initialise_random(n, m=None):
    """
    Initialise (n * m) shaped weights randomly.

    Parameters
    ----------
    n, m: int
        shape of the weight to initialise. (n * m)
        if m is none, weight is a 1d array of length = n

    Returns
    -------
    w: np.ndarray (n, m) float
    """
    # initialise weights in [-1/sqrt(n), 1/sqrt(n)) 
    # why? see below articles. 
    # https://leimao.github.io/blog/Weights-Initialization/
    # https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    limit = 1 / math.sqrt(n) if m is None else 1 / math.sqrt(n * m)
    size = n if m is None else (n, m)

    return np.random.uniform(-limit, limit, size)


def initialise_zero(n, m=None):
    """
    Initialise (n * m) shaped weights to zero. 

    Parameters
    ----------
    n, m: int
        shape of the weight to initialise. (n * m)
        if m is none, weight is a 1d array of length = n

    Returns
    -------
    w: np.ndarray (n, m) float
    """
    size = n if m is None else (n, m)
    return np.zeros(size)

###############################
# Hyperparameter Optimisation #
###############################

def auto_learning_rate(X):
    """
    Find good learning rate for square error minimisation. 
    opt learning rate = 1/p where p is max eigen value of X.T @ X
    Use Gershgorin circle theorem to estimate supremum of the 
    eiven vectors, instead of calculating actual max eigen value. 
    Ref(in JP): https://qiita.com/fujiisoup/items/e7f703fc57e2dfc441ad

    Parameters
    ----------
    X: np.array (n,d)
        feature matrix
        n: number of samples
        d: number of features
    """
    rho = supremum_eigen(X.T @ X)
    return 1.0 / rho
