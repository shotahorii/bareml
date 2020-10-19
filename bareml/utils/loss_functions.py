"""
Loss functions and Regularisations.

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

class SquareLoss:

    def __call__(self, X, y, w):
        return 0.5 * np.power(y - (X @ w), 2)

    def gradient(self, X, y, w):
        # X.T is a (d,n) array
        # (X @ w - y) is a (n,) array
        # X.T @ (X @ w - y) is a (d,) array
        return X.T @ (X @ w - y)

    def hessian(self, X, y, w):
        return np.ones(len(y))

class SquareError:

    def __call__(self, y, y_pred):
        return 0.5 * np.power(y - y_pred, 2)

    def gradient(self, y, y_pred):
        return y_pred - y

    def hessian(self, y, y_pred):
        return np.ones(len(y))

class CrossEntropy:

    def __call__(self, y, y_pred):
        """ 
        Calculates cross entropy. 
        This function applies for both binary classifiction (sigmoid)
        and multi class classification (softmax).
        
        Parameters
        ----------
        y: np.ndarray (int)
            one-hot encoded target variable 
            num of rows (y.shape[0]) is the num of samples 
            num of columns (y.shape[1]) is the num of classes
            each value is 0 or 1, where sum of values in a row is always 1.
            if it's a binary classification, this can also be 1d array.

        y_pred: np.ndarray (float)
            predicted probability of each class
            num of rows (y_pred.shape[0]) is the num of samples 
            num of columns (y_pred.shape[1]) is the num of classes
            each value is in [0,1], where sum of values in a row is always 1.
            if it's a binary classification, this can also be 1d array.
        """

        # Avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        if y.ndim == 1: # cross entropy for sigmoid (binary)
            return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        else: # cross entropy for softmax (multi classes)
            return - np.sum(y * np.log(y_pred), axis=1)
    
    def gradient(self, y, y_pred):
        """ 
        Calculates derivative of cross entropy. 
        This function applies for both binary classifiction (sigmoid)
        and multi class classification (softmax).
        
        Parameters
        ----------
        y: np.ndarray (int)
            one-hot encoded target variable 
            if it's a binary classification, this can also be 1d array.

        y_pred: np.ndarray (float)
            predicted probability of each class
            if it's a binary classification, this can also be 1d array.
        """
        return y_pred - y
        

class L1Regularization:
    """ L1 Regularization """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def gradient(self, w):
        raise ValueError('L1 loss is not differenciatable.')


class L2Regularization:
    """ L2 Regularization """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return 0.5 * self.alpha * np.dot(w.T, w)
    
    def gradient(self, w):
        return self.alpha * w