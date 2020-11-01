"""
Logistic Regression

Author: Shota Horii <sh.sinker@gmail.com>
Test: tests/test_logistic_regresion.py

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 205-220.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 247-263.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 119-127.
Y. Yajima et al. (1992). 自然科学の統計学. 東京大学出版会. 231-249.
T. Kubo (2012). データ解析のための統計モデリング入門. 岩波書店. 114-127.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 88-95.
H. Toyoda (2017). 実践ベイズモデリング. 朝倉書店. 64-66.
"""


import math
import numpy as np

from ..base import Classifier
from ..utils.activation_functions import Sigmoid, Softmax
from ..utils.manipulators import add_intercept, prob2binary, StandardScaler
from ..utils.optimise import GradientDescent

class LogisticRegression(Classifier):
    """ 
    Logistic Regression classifier
    
    Parameters
    ----------
    fit_intercept: bool
        add intercept [1,1,1,1,...] column to X if True

    C: float > 0
        Inverse of regularization strength. 
        Smaller means stronger regularisation.

    max_iter: int > 0
        number of iterations that we update w with gradient descent

    tol: float >= 0
        Conversion criterion. Applicable to solvers using iterative method.
        In each step, if delta is smaller than tol, algorighm considers it's converged.

    lr: float > 0
        learning rate multiplied to gradient in each updation
    """

    def __init__(self, fit_intercept=True,
        C=1.0, max_iter=1000, tol=1e-4, lr=None):
        
        self.fit_intercept = fit_intercept
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.w = None

    def _fit(self, X, y):
        """ 
        Train the logistic regression model.
        
        Parameters
        ----------
        X: np.ndarray (n, d) 
            n: number of samples
            d: number of features
        
        y: np.ndarray (n, c)
            n: number of samples
            c: number of classes: c=2 when binary

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

        self.scaler = StandardScaler()
        if self.fit_intercept:
            X[:,1:] = self.scaler.fit(X[:,1:]).transform(X[:,1:])
        else:
            X = self.scaler.fit(X).transform(X)

        # note: input y is always shape of (n,c)
        # even if it's binary classification, it's (n,2) not (n,)
        # see implementation of bareml.base.Classifier
        if y.shape[1] == 2: # binary classification
            y = y[:,1]
            self.activation = Sigmoid()
        else:
            self.activation = Softmax()

        # function to calculate gradient of loss function w.r.t. w
        def gradient(X, y, w):
            # X.T is a (d,n) array
            # (X @ w - y) is a (n,c) array if multi-class
            #                a (n,) array if binary
            # w & penalty is a (d,c) array if multi-class
            #                a (d,) array if binary
            # X.T @ (X @ w - y) + self.alpha * w is a (d,c) array if multi-class
            #                                       a (d,) array if binary
            if self.fit_intercept:
                penalty = np.insert(w[1:], 0, 0, axis=0) # no penalise intercept
            else:
                penalty = w
            return self.C * X.T @ (self.activation(X @ w) - y) + penalty

        # initialise optimiser
        opt = GradientDescent(
            gradient=gradient, max_iter=self.max_iter,
            tol=self.tol, lr=self.lr)
        
        # optimise
        self.w = opt.solve(X, y)

        return self

    def _predict_proba(self, X):
        """ 
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

        if self.fit_intercept:
            X[:,1:] = self.scaler.transform(X[:,1:])
        else:
            X = self.scaler.transform(X)

        pred = self.activation(np.dot(X,self.w))

        if pred.ndim == 1: # binary classification
            # convert to (n,2) shape from (n,) shape
            pred = np.vstack([1-pred, pred]).T

        return pred

    def _predict(self, X):
        y_pred = prob2binary(self._predict_proba(X))
        return y_pred
