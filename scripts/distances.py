import math
import numpy as np

def euclidean_distance(v1, v2):
    """ 
    Calculate euclidean distance (l2 distance) between two vectors 

    Parameters
    ----------
    v1, v2: np.array

    Returns
    -------
    float
    """
    if len(v1) != len(v2):
        raise ValueError("2 vectors must have same dimension")

    return math.sqrt(np.power(v1 - v2, 2).sum())

