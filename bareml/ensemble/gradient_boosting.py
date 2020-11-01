"""
Gradient Boosting

Author: Shota Horii <sh.sinker@gmail.com>

References:
[1] K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 545-.
[2] P. Buhlmann and T.Hothorn (2007). 
    Boosting Algorithms: Regularization, Prediction and Model Fitting.
    Statistical Science 22(4), 477-505. (https://arxiv.org/pdf/0804.2752.pdf)

ToDo:
Abstract classes for 
- AdaptiveBasisFunctionModel
- ForwardStagewiseAdditiveModeling
"""

from abc import ABC, abstractmethod
import numpy as np

from ..base import Regressor, Ensemble
from ..supervised.decision_trees import DecisionTreeRegressor
from ..utils.loss_functions import SquareError

class GradientBoosting(Ensemble):
    
    def __init__(self, estimator, loss, max_iter=10, lr=0.1):
        super().__init__(base_estimator=estimator)
        self.loss = loss
        self.max_iter = max_iter
        self.lr=lr
        self.y_mean = None

    def _f0(self, X):
        return np.full(len(X), self.y_mean)

    def _fit(self, X, y):
        
        # init
        self.y_mean = y.mean(axis=0)
        y_pred = self._f0(X)

        for m in range(self.max_iter):
            print(m)
            # compute the gradient residual
            r = (-1.0) * self.loss.gradient(y, y_pred)
            # fit the weak learner
            est = self._make_estimator()
            y_pred = y_pred + self.lr * est.fit(X, r).predict(X)

    def _predict(self, X):
        return self._f0(X) + sum([self.lr * est.predict(X) for est in self.estimators])
    

class L2Boosting(GradientBoosting, Regressor):
    
    def __init__(self, estimator=DecisionTreeRegressor(max_depth=1), max_iter=10, lr=0.1):
        super().__init__(estimator=estimator, loss=SquareError(), max_iter=max_iter, lr=lr)

