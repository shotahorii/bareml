"""
Preprocessing

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
from itertools import combinations_with_replacement
import numpy as np

from scripts.util import flatten


def binarise(X, threshold=0.5):
    """
    Binarises values in array

    Parameters
    ----------
    X: np.array (any shape) float/int in (-inf, inf)

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

def add_bias(X):
    """
    Add bias (a column filled with 1. ) to the feature matrix.

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

def normalise(X):
    pass

def standardise(X):
    pass