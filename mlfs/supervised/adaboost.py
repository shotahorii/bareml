"""
Bagging

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
from mlfs.utils.misc import prob2binary
from mlfs.supervised.base_classes import Classifier, Regressor

class AdaBoost(Classifier):
    """
    AdaBoost Classifier (Binary classification only)
    """
    pass