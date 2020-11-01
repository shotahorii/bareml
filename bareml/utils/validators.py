"""
Validation

References:

ToDo: GroupKFold
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from .misc import split_array
from .manipulators import OnehotEncoder


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


def cross_val_predict(estimator, X, y=None, cv=5, stratified=False, shuffle=False, seed=None):

    kf = StratifiedKFold(cv, shuffle, seed) if stratified else KFold(cv, shuffle, seed)

    test_indices = []
    test_preds = []

    for train_idx, test_idx in kf.split(X, y):

        X_train = X[train_idx]
        y_train = None if y is None else y[train_idx]

        test_pred = estimator.fit(X_train, y_train).predict(X[test_idx])

        test_indices.append(test_idx)
        test_preds.append(test_pred)

    test_indices = np.concatenate(test_indices)
    test_preds = np.concatenate(test_preds)

    return test_preds[test_indices.argsort()]
    

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
        

class StratifiedKFold:
    """
    Stratified KFold cross validation.

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

        if y.ndim != 1: # multi-class classification
            onehot = OnehotEncoder()
            y = onehot.inverse_transform(y) # express in 1d array

        classes = np.unique(y) # list of classes

        # divide entire indices into each class
        # e.g. if y = [0,0,1,2,2] -> by_class = [[0,1],[2],[3,4]]
        by_class = [np.where(y==c)[0] for c in classes]

        # shuffle each class's index list, if shuffle.
        if self.shuffle:
            if self.seed:
                np.random.seed(self.seed)
            for l in by_class:
                np.random.shuffle(l)

        # split each class's index list into k-chunks
        by_class_splitted = [list(split_array(l,self.n_splits)) for l in by_class]
        # merge all classes in same chunk
        chunks = [np.concatenate([by_class_splitted[c][s] for c in range(len(classes))]) for s in range(self.n_splits)]

        for test_idx in chunks:
            train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
            yield train_idx, test_idx


class GroupKFold:
    """
    K-fold iterator variant with non-overlapping groups.
    The same group will not appear in two different folds 
    (the number of distinct groups has to be at least equal to the number of folds).

    Parameters
    ----------
    n_splits: int >= 2
        Number of folds
    """
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def split(self, X, y, groups):
        """
        Parameters
        ----------
        X: np.ndarray (n, d)
        y: np.ndarray (n, c)
        groups np.ndarray (n,)

        Yields
        ------
        train_idx: np.ndarray (a,)
        test_idx: np.ndarray (n-a,)
        """
        pass


    
