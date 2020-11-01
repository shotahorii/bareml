"""
Voting

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
from abc import ABC, abstractmethod

from ..base import Classifier, Regressor, Ensemble

class Voting(Ensemble):

    def __init__(self, estimators, voting='hard', weights=None):
        super().__init__(estimators=estimators)