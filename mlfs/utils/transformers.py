"""
Transformer functions and classes

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
from itertools import combinations_with_replacement
import numpy as np

from mlfs.utils.misc import flatten


def real2binary(X, threshold=0.5):
    """
    Convert real values (-inf, inf) -> binary values {0,1}

    Parameters
    ----------
    X: np.array (n, m) float/int in (-inf, inf)

    threshold: float in (-inf, inf)
        value greater than this is converted into 1, otherwise 0
    """
    return (X > threshold).astype(int)

def encode_onehot(X):
    """ 
    Encodes nominal values to one-hot encoding 
    
    Parameters
    ----------
    X: np.array (n,)
        n: number of samples 
    """
    uniq = np.unique(X)
    return np.array([(X==uv).astype(int) for uv in uniq]).T   

def decode_onehot(X):
    """ 
    Decodes one-hot encoding to nominal values 

    Parameters
    ----------
    X: np.array (n, c) int/float in {0, 1}
        n: number of samples 
        c: number of classes
    """
    return np.argmax(X, axis=1)

def add_intercept(X):
    """
    Add intercept (a column filled with 1. ) to the feature matrix.

    Parameters
    ----------
    X: np.array (n, d) float in (-inf, inf)
        n: number of samples 
        d: number of features
    """
    return np.insert(X, 0, 1, axis=1)

def polynomial_features(X, degree):
    """
    Generates a new feature matrix consisting of all polynomial combinations 
    of the features with degree less than or equal to the given degree. 
    e.g. X=[a, b], degree=2 then [1, a, b, a^2, ab, b^2]

    Parameters
    ----------
    X: np.array (n, d) float in (-inf, inf)
        n: number of samples 
        d: number of features

    degree: int in {1,2,3,...} 
    """
    n_features = X.shape[1]
    index_combinations = flatten([combinations_with_replacement(range(n_features), i) for i in range(0,degree+1)])
    
    return np.array([np.prod(X[:, comb], axis=1) for comb in index_combinations]).T  


class StandardScaler:
    """ 
    Feature Scaler (Standardisation)
    X -> (X - mu) / std 
    """

    def __init__(self):
        self.mu = None
        self.s = None
    
    def fit(self, X):
        """
        Compute and store mean and standard deviation 
        for each column (feature) of the given matrix X.

        Parameters
        ----------
        X: np.array (n, d) float in (-inf, inf)
            n: number of samples 
            d: number of features
        """
        self.mu = np.mean(X, axis=0)
        self.s = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Transform the given matrix X using the stored mean and std. 
        X -> (X - mu) / std

        Parameters
        ----------
        X: np.array (n, d) float in (-inf, inf)
            n: number of samples 
            d: number of features
        """
        return (X - self.mu) / self.s

class OnehotEncoder:
    pass


def prob2binary(y):
    """
    Convert probability to binary data. 
    For example, [0.6, 0.2, 0.8] -> [1, 0, 1]
    Also, [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]] -> [[0, 1, 0], [0, 0, 1]]

    Parameters
    ----------
    y: np.ndarray (n,d)
    """
    if y.ndim == 1:
        return np.round(y).astype(int)
    else:
        # avoid [[0.333, 0.333, 0.333], [0.2, 0.4, 0.4]] -> [[1, 1, 1], [0, 1, 1]]
        # instead [[0.333, 0.333, 0.333], [0.2, 0.4, 0.4]] -> [[1, 0, 0], [0, 1, 0]]
        y_bin = np.zeros_like(y)
        y_bin[np.arange(len(y)), y.argmax(axis=1)] = 1 
        return y_bin
        
        # random pick
        #while True:
        #    y_plus_r = y + 1e-15 * np.random.rand(y.shape[0],y.shape[1])
        #    binary = (y_plus_r == y_plus_r.max(axis=1)[:,None]).astype(int)
        #    if binary.sum() == len(y):
        #        return binary
        

def binary2onehot(y):
    """
    Convert binary to one-hot expression. 
    e.g. np.array([1,0,0,1]) -> np.array([[0,1], [1,0], [1,0], [0,1]])

    Parameters
    ----------
    y: np.array (n,) int {0,1}

    Returns
    -------
    np.array (n,2)
    """
    return np.array([y, (y!=1).astype(int)]).T