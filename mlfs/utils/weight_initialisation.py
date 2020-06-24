"""
Weight initialisation methods

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

def initialise_random(n_weights, n_classes=None):
    """
    Initialise weights randomly.

    Parameters
    ----------
    n_weights: int
        number of weights to initialise

    n_classes: int

    Returns
    -------
    w: np.ndarray (d,) float
        d: number of weights
    """
    # initialise weights in [-1/sqrt(n), 1/sqrt(n)) 
    # why? see below articles. 
    # https://leimao.github.io/blog/Weights-Initialization/
    # https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    limit = 1 / math.sqrt(n_weights)
    if n_classes is None:
        w = np.random.uniform(-limit, limit, n_weights)
    else:
        w = np.random.uniform(-limit, limit, (n_weights, n_classes))
    return w

def initialise_zero(n_weights, n_classes=None):
    """
    Initialise weights to zero. 

    Parameters
    ----------
    n_weights: int
        number of weights to initialise
    
    n_classes: int
    
    Returns
    -------
    w: np.ndarray (d,) float
        d: number of weights
    """
    if n_classes is None:
        w = np.zeros(n_weights)
    else:
        w = np.zeros((n_weights, n_classes))
    return w