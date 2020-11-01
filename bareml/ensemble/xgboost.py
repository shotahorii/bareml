"""
XGBoost

Author: Shota Horii <sh.sinker@gmail.com>

References:
XGBoost: A Scalable Tree Boosting System (https://arxiv.org/pdf/1603.02754.pdf)
https://xgboost.readthedocs.io/en/latest/tutorials/model.html
https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf

"""

import math
import numpy as np
from abc import ABC, abstractmethod

from ..base import Classifier, Regressor, Ensemble
from ..utils.loss_functions import SquareError
from ..supervised.decision_trees import GBTree

class XGBoost(Ensemble, Regressor):

    def __init__(self, loss=SquareError(), max_iter=10, lr=0.3):
        super().__init__(base_estimator=GBTree())
        self.loss = loss
        self.max_iter = max_iter
        self.lr=lr

    def _fit(self, X, y):

        # init
        y_pred = np.zeros(len(y))

        for m in range(self.max_iter):
            print(m)
            # compute the gradient & hessian
            g = self.loss.gradient(y, y_pred)
            h = self.loss.hessian(y, y_pred)
            y_tmp = np.array([g,h]).T
            # fit the weak learner
            est = self._make_estimator()
            y_pred = y_pred + self.lr * est.fit(X, y_tmp).predict(X)

    def _predict(self, X):
        return sum([self.lr * est.predict(X) for est in self.estimators])
