"""
base classes
"""

# Author: Shota Horii <sh.sinker@gmail.com>

from abc import ABC

class Classifier(ABC):
    pass

class Regressor(ABC):
    pass

class Weighted(ABC):
    """ 
    Weighted samples are used for training.
    fit() takes X, y and weights as parameters.
    """
    pass