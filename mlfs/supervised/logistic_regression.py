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

ToDo:
- Regularization
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from mlfs.utils.activation_functions import Sigmoid, Softmax
from mlfs.utils.preprocessing import add_bias, polynomial_features, StandardScaler
from mlfs.utils.solvers import CrossEntropyGD, CrossEntropyMultiGD
from mlfs.utils.misc import prob2binary
from mlfs.supervised.base_classes import Classifier

class LogisticRegression(Classifier):
    """ 
    Ligistic Regression classifier
    """

    def __init__(self, multiclass=False, alpha_l2=0, polynomial_degree=1, 
        max_iterations=1000, tol=1e-4, learning_rate=None):
        
        self.alpha_l2 = alpha_l2
        self.polynomial_degree = polynomial_degree
        self.max_iterations = max_iterations
        self.tol = tol
        self.learning_rate = learning_rate
        self.w = None
        self.train_error = None

        self.activation = Softmax() if multiclass else Sigmoid()
        self.solver = CrossEntropyMultiGD if multiclass else CrossEntropyGD

        self.scaler = StandardScaler()

    def fit(self, X, y):
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

        n_iterations: int
            number of iterations that we update w with gradient descent

        learning_rate: float
            learning rate multiplied to gradient in each updation

        Returns
        -------
        self: LogisticRegression
        """

        X = polynomial_features(X, self.polynomial_degree)
        X[:,1:] = self.scaler.fit(X[:,1:]).transform(X[:,1:])

        gd = self.solver(self.alpha_l2, self.max_iterations, self.tol, self.learning_rate)
        self.w = gd.solve(X,y)

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
        X = polynomial_features(X, self.polynomial_degree)
        X[:,1:] = self.scaler.transform(X[:,1:])

        y_pred = self.activation(np.dot(X,self.w))

        return prob2binary(y_pred)
