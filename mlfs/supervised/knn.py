"""
K Nearest Neighbors

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 124-127.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 511-512.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 463-481.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 54-69.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import numpy as np
from abc import ABC, abstractmethod

from mlfs.utils.distances import euclidean_distance
from mlfs.supervised.base_classes import Classifier, Regressor
from mlfs.utils.transformers import prob2binary

class _KNN(ABC):
    """ 
    K Nearest Neighbors

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

    def fit(self, X, y):
        """ 
        Just store the training data. Nothing else.
        
        Parameters
        ----------
        X: np.ndarray
            predictor variables 
            num of rows (X.shape[0]) is the num of samples 
            num of columns (X.shape[1]) is the num of variables
        
        y: np.ndarray
            one-hot encoded target variable 
            num of rows (y.shape[0]) is the num of samples 
            num of columns (y.shape[1]) is the num of classes
            each value is 0 or 1, where sum of values in a row is always 1

        Returns
        -------
        self: KNNClassifier
        """
        self.X = X
        self.y = y

        return self 

    @abstractmethod
    def _predict(self, k_nearest_y):
        pass

    def predict(self, X):
        """ 
        Parameters
        ----------
        X: np.ndarray
            predictor variables of the samples that we wish to predict

        Returns
        -------
        y_pred: np.ndarray
            one-hot encoded target variable
        """

        if self.y.ndim == 1: # regression or binary classification
            y_preds = np.zeros([X.shape[0])
        else: # multi class classification
            y_preds = np.zeros([X.shape[0],self.y.shape[1]])

        for i, x_pred in enumerate(X):
            # find k nearest trained data
            k_nearest_idx = np.argsort([euclidean_distance(x_pred, x) for x in self.X])[:self.k]
            # get target variables of those k data
            k_nearest_y = self.y[k_nearest_idx]
            # update y_pred
            y_preds[i] = self._predict(k_nearest_y)
            
        return y_preds
        

class KNNClassifier(_KNN, Classifier):
    """ 
    K Nearest Neighbors classifier

    Parameters
    ----------
    k: int
        The number of nearest data points that we use to 
        determine the class of the sample to predict.
    """

    def __init__(self, k=1):
        super().__init__(k)

    def fit(self, X, y):
        X, y = self._validate_Xy(X, y)
        return super().fit(X, y)

    def _predict(self, k_nearest_y):
        return k_nearest_y.sum(axis=0) / len(k_nearest_y)

    def predict_proba(self, X):
        X = self._validate_X(X)
        return super().predict(X)

    def predict(self, X):
        X = self._validate_X(X)
        return prob2binary(super().predict(X))


class KNNRegressor(_KNN, Regressor):
    """ 
    K Nearest Neighbors regressor

    Parameters
    ----------
    k: int
        The number of nearest data points that we use to 
        determine the class of the sample to predict.
    """

    def __init__(self, k=1):
        super().__init__(k)

    def fit(self, X, y):
        X, y = self._validate_Xy(X, y)
        return super().fit(X, y)

    def _predict(self, k_nearest_y):
        return np.mean(k_nearest_y)

    def predict(self, X):
        X = self._validate_X(X)
        return super().predict(X)