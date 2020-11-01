"""
Sampling methods

Author: Shota Horii <sh.sinker@gmail.com>

References:
"""


import math
import random
import numpy as np


def bootstrap_sampling(X, y, sampling_ratio=1.0):
    """ bootstrap sampling """
    n_samples = int(round(len(X) * sampling_ratio))
    sample_idx = random.choices( np.arange(len(X)), k=n_samples)
    return X[sample_idx], y[sample_idx]