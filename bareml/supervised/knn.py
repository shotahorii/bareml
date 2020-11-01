"""
K Nearest Neighbors

Author: Shota Horii <sh.sinker@gmail.com>
Test: tests/test_knn.py

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 124-127.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 511-512.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 463-481.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 54-69.
"""

import numpy as np

from ..base import Classifier, Regressor
from ..utils.distances import euclidean_distance
from ..utils.manipulators import prob2binary

class KNNClassifier(Classifier):
    """ 
    KNN Classifier

    Parameters
    ----------
    k: int
        The number of nearest data points that we use to 
        determine the class of the sample to predict.
    """

    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None

    def _fit(self, X, y):
        """ 
        Parameters
        ----------
        X: np.array (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features
        
        y: np.ndarray (n, c) of int {0,1}
            n: number of samples
            c: number of classes
        """
        self.X = X
        self.y = y

        return self 

    def _predict(self, X):
        """ 
        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)

        Returns
        -------
        y_preds: np.ndarray (n, c) of int {0,1}
        """
        return prob2binary(self._predict_proba(X))

    def _predict_proba(self, X):
        """ 
        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)

        Returns
        -------
        y_preds: np.ndarray (n, c) of real [0,1]
        """

        if self.y.ndim == 1: # binary classification
            y_preds = np.zeros(X.shape[0])
        else: # multi class classification
            y_preds = np.zeros([X.shape[0],self.y.shape[1]])

        for i, x_pred in enumerate(X):
            # find k nearest trained data
            k_nearest_idx = np.argsort([euclidean_distance(x_pred, x) for x in self.X])[:self.k]
            # get target variables of those k data
            k_nearest_y = self.y[k_nearest_idx]
            # update y_pred
            y_preds[i] = k_nearest_y.sum(axis=0) / len(k_nearest_y)
            
        return y_preds


class KNNRegressor(Regressor):
    """ 
    KNN Regressor

    Parameters
    ----------
    k: int
        The number of nearest data points that we use to 
        determine the class of the sample to predict.
    """

    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None

    def _fit(self, X, y):
        """ 
        Parameters
        ----------
        X: np.array (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features
        
        y: np.ndarray (n,) of real (-inf, inf)
            n: number of samples
        """
        self.X = X
        self.y = y

        return self 

    def _predict(self, X):
        """ 
        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)

        Returns
        -------
        y_preds: np.ndarray (n,) of real (-inf, inf)
        """

        y_preds = np.zeros(X.shape[0])

        for i, x_pred in enumerate(X):
            # find k nearest trained data
            k_nearest_idx = np.argsort([euclidean_distance(x_pred, x) for x in self.X])[:self.k]
            # get target variables of those k data
            k_nearest_y = self.y[k_nearest_idx]
            # update y_pred
            y_preds[i] = np.mean(k_nearest_y)
            
        return y_preds