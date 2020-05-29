"""
Loss functions.

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import numpy as np

class CrossEntropy:

    def __call__(self, y, y_pred):

        # Avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        if y.ndim == 1: # cross entropy for sigmoid (binary classification)
            return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        else: # cross entropy for softmax (multi classes classification)
            return - np.sum(y * np.log(y_pred), axis=1)
    
    def gradient(self, y, y_pred):
        return y_pred - y