"""
Loss functions.

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import numpy as np

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

        if y.ndim == 1: # cross entropy for sigmoid (binary classification)
            return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        else: # cross entropy for softmax (multi classes classification)
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
        