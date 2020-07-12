"""
Perceptron. 
A simple implementation of Perceptron classifier algorithm
described in Bishop(2006).

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 192-196.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import numpy as np

from mlfs.supervised.base_classes import Classifier

class Perceptron(Classifier):
    """
    Perceptron Classifier (Binary classification only)
    """

    def __init__(self):
        self.w = None
        self.b = None

    def _sign(self, vals):
        """
        Returns -1 for values < 0, 1 for values >= 0.
        Similar to np.sign but does not return 0 for value == 0.

        Parameters
        ----------
        vals: np.array (n,)

        Returns 
        -------
        np.array (n,)
        """
        return np.array([-1 if v < 0 else 1 for v in vals])

    def fit(self, X, y, max_iterations=500):
        """
        Parameters
        ----------
        X: np.array (n,d)
            feature variables
            n: number of samples
            d: number of features
        
        y: np.array (n,)
            target variable (binary classification)
            n: number of samples
            all elements in {0,1} or {-1,1}

        max_iterations: int
            maximum number of iterations of the weight update steps

        Returns 
        -------
        self: Perceptron
        """

        # convert target variable from {0,1} -> {-1, 1}
        y = np.array([-1 if v == 0 else v for v in y])

        # initialise the weights and bias
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        for _ in range(max_iterations):

            y_pred = self._sign(self.w @ X.T + self.b)
            X_misclassified = X[y_pred != y]
            y_misclassified = y[y_pred != y]

            if len(X_misclassified) == 0:
                break
            
            # SGD
            idx = np.random.choice(len(X_misclassified)) # pick one data point
            self.w += X_misclassified[idx] * y_misclassified[idx]
            self.b += y_misclassified[idx]

        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X: np.array (n,d)
            feature variables to be used for the prediction
            n: number of samples
            d: number of features

        Returns 
        -------
        np.ndarray (n,)
            predicted classes
            n: number of samples
            all elements in {-1,1}
        """
        return self._sign(self.w @ X.T + self.b)
