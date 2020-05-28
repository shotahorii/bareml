"""
Logistic Regression

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 205-220.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 247-263.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 119-127.
Y. Yajima et al. (1992). 自然科学の統計学. 東京大学出版会. 231-249.
T. Kubo (2012). データ解析のための統計モデリング入門. 岩波書店. 114-127.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 88-95.
H. Toyoda (2017). 実践ベイズモデリング. 朝倉書店. 64-66.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from scripts.activation_functions import Sigmoid
from scripts.loss_functions import CrossEntropy

class LogisticRegressionClassifer:
    """ 
    Ligistic Regression classifier (Binary Classification only)
    """

    def __init__(self):
        self.w = None
        self.b = None

        self.sigmoid = Sigmoid()
        self.cross_entropy = CrossEntropy()

    def _initialise_weights_bias(self, n_features):
        """
        Initialise the value of weights and bias.

        Parameters
        ----------
        n_features: int
            number of predictor variables in input data X
        """

        # initialise weights in [-1/sqrt(n), 1/sqrt(n)) 
        # why? see below articles. 
        # https://leimao.github.io/blog/Weights-Initialization/
        # https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, n_features)
        self.b = 0.0

    def fit(self, X, y, n_iterations=2000, learning_rate=0.1):
        """ 
        Train the logistic regression model. Binary classification only. 
        
        Parameters
        ----------
        X: np.ndarray
            predictor variables 
            num of rows (X_train.shape[0]) is the num of samples 
            num of columns (X_train.shape[1]) is the num of variables
        
        y: np.ndarray (1d array)
            target variable. As this is a binary classification, 
            y is a 1d-array with value of 0 or 1. 
            num of elements (y_train.shape[0]) is the num of samples 

        Returns
        -------
        self: LogisticRegressionClassifer
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self._initialise_weights_bias(n_features)

        for _ in range(n_iterations):

            y_pred = self.sigmoid(np.dot(self.w, X.T)+self.b)
            # loss = self.cross_entropy(y, y_pred) # no need to calculate loss in every iteration
            loss_grad = self.cross_entropy.gradient(y, y_pred)
            
            # calculate gradient for weights and bias
            w_grad = np.dot(X.T, loss_grad.T)/n_samples
            b_grad = np.sum(loss_grad)/n_samples
            # update weights and bias
            self.w = self.w - learning_rate * w_grad
            self.b = self.b - learning_rate * b_grad

        return self

    def predict(self, X):
        """ 
        Predict.
        
        Parameters
        ----------
        X: np.ndarray
            predictor variables of the samples you wish to predict.
        
        Returns
        -------
        y_pred: np.ndarray (1d array)
            target variable. 
        """
        y_pred = np.round(self.sigmoid(np.dot(self.w, X.T)+self.b)).astype(int)
        return y_pred

            
