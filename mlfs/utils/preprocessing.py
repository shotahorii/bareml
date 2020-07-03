"""
Preprocessing

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
from itertools import combinations_with_replacement
import numpy as np

from mlfs.utils.misc import flatten


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


def shuffle_data(X, y, seed=None):
    """ 
    Random shuffle of the samples in X and y 
    https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/utils/data_manipulation.py
    """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_ratio=0.5, shuffle=True, seed=None):
    """ 
    Split the data into train and test sets 
    https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/utils/data_manipulation.py
    """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_ratio))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test