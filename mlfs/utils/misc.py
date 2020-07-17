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
