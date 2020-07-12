"""
Validation

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math

import numpy as np

from mlfs.utils.misc import split_array


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


class KFold:
    """
    KFold cross validation.

    Parameters
    ----------
    n_splits: int >= 2
        Number of folds
    shuffle: bool
        Whether to shuffle the data before splitting into batches
    seed: int
        random state.
    """
    def __init__(self, n_splits=5, shuffle=False, seed=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X, y):
        indices = np.arange(len(X))
        if self.shuffle:
            if self.seed:
                np.random.seed(self.seed)
            np.random.shuffle(indices)

        for test_idx in split_array(indices, self.n_splits):
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx
        
