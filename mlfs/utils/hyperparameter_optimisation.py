"""
Hyperparameter optimisation

References:


ToDo:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from mlfs.utils.misc import supremum_eigen

def auto_learning_rate_se(X):
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
