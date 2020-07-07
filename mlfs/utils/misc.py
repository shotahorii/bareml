"""
Utility functions

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import operator as op
from functools import reduce
import math
import numpy as np

def ncr(n, r):
    """
    Calculates nCr in efficient manner. 
    This function is not my original code, but copied from below url.
    https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python

    Parameters
    ----------
    n, r: int 
    """
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def flatten(l):
    """ 
    flatten a nested list.
    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    
    Parameters
    ----------
    l: array-like
    """
    return [item for sublist in l for item in sublist]


def supremum_eigen(X):
    """
    Estimates approximate supremum of eigen values of a square matrix
    by Gershgorin circle theorem.
    Ref(in JP): https://qiita.com/fujiisoup/items/e7f703fc57e2dfc441ad

    Parameters
    ----------
    X: np.ndarray (d,d)
    """
    return np.max(np.sum(np.abs(X), axis=0))


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

def split_array(a, n):
    """
    Split an array into n chunks.
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

    Parameters
    ----------
    a: array-like
    n: int 
        number of chunks
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))