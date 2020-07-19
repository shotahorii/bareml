"""
Voting

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
from abc import ABC, abstractmethod

from mlfs.supervised.base_classes import Classifier, Regressor

class Voting:

    def __init__(self, estimators, voting='hard', weights=None):
        pass