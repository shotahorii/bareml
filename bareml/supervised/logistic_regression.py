"""
Logistic Regression

Author: Shota Horii <sh.sinker@gmail.com>

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 205-220.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 247-263.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 119-127.
Y. Yajima et al. (1992). 自然科学の統計学. 東京大学出版会. 231-249.
T. Kubo (2012). データ解析のための統計モデリング入門. 岩波書店. 114-127.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 88-95.
H. Toyoda (2017). 実践ベイズモデリング. 朝倉書店. 64-66.

ToDo:
- Regularization
"""


import math
import numpy as np

from bareml import Classifier
from bareml.utils.activation_functions import Sigmoid, Softmax
from bareml.utils.manipulators import add_intercept, prob2binary
from bareml.utils.solvers import CrossEntropyGD, CrossEntropyMultiGD

class LogisticRegression(Classifier):
    """ 
    Ligistic Regression classifier
    """

    def __init__(self, multiclass=False, fit_intercept=True,
        alpha_l2=0, max_iter=1000, tol=1e-4, lr=None):
        
        self.fit_intercept = fit_intercept
        self.alpha_l2 = alpha_l2
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.w = None

        self.activation = Softmax() if multiclass else Sigmoid()
        self.solver = CrossEntropyMultiGD if multiclass else CrossEntropyGD

    def _fit(self, X, y):
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

        max_iter: int
            number of iterations that we update w with gradient descent

        lr: float
            learning rate multiplied to gradient in each updation

        Returns
        -------
        self: LogisticRegression
        """

        if self.fit_intercept:
            X = add_intercept(X)

        gd = self.solver(self.alpha_l2, self.max_iter, self.tol, self.lr)
        self.w = gd.solve(X,y)

        return self

    def _predict_proba(self, X):
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
        if self.fit_intercept:
            X = add_intercept(X)

        return self.activation(np.dot(X,self.w))

    def _predict(self, X):
        return prob2binary(self._predict_proba(X))
