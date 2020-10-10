"""
Perceptron. 
A simple implementation of Perceptron classifier algorithm
described in Bishop(2006).

Author: Shota Horii <sh.sinker@gmail.com>
Test: tests/test_perceptron.py

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 192-196.
"""

import numpy as np

from bareml import BinaryClassifier
from bareml.utils.manipulators import binary2sign, real2sign, real2binary


class Perceptron(BinaryClassifier):
    """
    Perceptron Classifier (Binary classification only)

    Parameters
    ----------
    n_epoch: int
        number of epoch to iterate optimisation process
    
    shuffle: bool
        if true, shuffle data before optimisation

    seed: int
        random seed
    """

    def __init__(self, n_epoch=10, shuffle=True, seed=0):
        self.w = None
        self.b = None
        self.n_epoch = n_epoch
        self.shuffle = shuffle
        self.seed = seed

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features
        
        y: np.ndarray (n,) of int {0,1}
            n: number of samples 

        Returns 
        -------
        self: Perceptron
        """
        # validate the input data
        X, y = self._validate_Xy(X, y)

        # convert target variable from {0,1} -> {-1, 1}
        y = binary2sign(y)

        # initialise the weights and bias
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        
        idx = np.arange(X.shape[0])
        # shuffle data
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(idx)

        # SGD
        for _ in range(self.n_epoch):
            has_misclassification = False
            for i in idx:
                y_pred = real2sign(self.w @ X[i] + self.b)
                if y_pred != y[i]:
                    has_misclassification = True
                    self.w += X[i] * y[i]
                    self.b += y[i]
            # terminate if all samples were classified correctly
            if not has_misclassification:
                break

        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features

        Returns 
        -------
        np.ndarray (n,) of int {0,1}
            n: number of samples
        """
        X = self._validate_X(X)
        return real2binary(self.w @ X.T + self.b, threshold=0)
