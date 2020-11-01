"""
Distance Metrics 

Author: Shota Horii <sh.sinker@gmail.com>

References:
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 153-154.
"""


import math
import numpy as np

def minkowski_distance(p, v1, v2):
    """ 
    Calculates minkowski distance (lp distance) between two vectors 

    Parameters
    ----------
    p: float
        parameter of minkowski distance
        when p=1, manhattan distance
        when p=2, euclidean distance

    v1, v2: np.array
        vectors to calculate distance between

    Returns
    -------
    distance: float
    """
    if len(v1) != len(v2):
        raise ValueError("2 vectors must have same dimension")
    
    # aka norm(p,v1-v2)
    distance = np.power(np.sum(np.power(np.abs(v1-v2),p)),1/p)
    return distance

def euclidean_distance(v1, v2):
    """ 
    Calculates euclidean distance (l2 distance) between two vectors 

    Parameters
    ----------
    v1, v2: np.array
        vectors to calculate distance between

    Returns
    -------
    distance: float
    """
    # aka minkowski_distance(2, v1, v2)
    # aka l2_norm(vi - v2)
    distance = math.sqrt(np.power(v1 - v2, 2).sum())
    return distance

def manhattan_distance(v1, v2):
    """ 
    Calculates manhattan distance (l1 distance) between two vectors 

    Parameters
    ----------
    v1, v2: np.array
        vectors to calculate distance between

    Returns
    -------
    distance: float
    """
    # aka minkowski_distance(1, v1, v2)
    # aka l1_norm(vi - v2)
    distance = np.abs(v1 - v2).sum()
    return distance

def chebyshev_distance(v1, v2):
    """ 
    Calculates chebyshev distance (l inf distance) between two vectors 

    Parameters
    ----------
    v1, v2: np.array

    Returns
    -------
    distance: float
    """
    if len(v1) != len(v2):
        raise ValueError("2 vectors must have same dimension")

    # aka lim p->infinity{ minkowski_distance(p, v1, v2) }
    # aka sup_norm(v1 - v2)
    distance = max(np.abs(v1 - v2))
    return distance

def norm(p, x):
    return np.power(np.sum(np.power(np.abs(x),p)),1/p)

def l1_norm(x):
    return np.abs(x).sum()

def l2_norm(x):
    return math.sqrt(np.power(x,2).sum())

def sup_norm(x):
    return max(np.abs(x))


def point_hyperplane_distance(x, w, b):
    """
    Distance from a point x to a hyperplane wTx + b
    https://math.stackexchange.com/questions/1210545/distance-from-a-point-to-a-hyperplane
    """
    return np.abs(x @ w + b) / l2_norm(w)
