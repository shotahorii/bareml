"""
Bagging

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 657-663.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
from mlfs.utils.misc import prob2binary
from mlfs.supervised.base_classes import Classifier, Regressor
from mlfs.supervised.decision_trees import WeightedDecisionStump

class AdaBoost(Classifier):
    """
    AdaBoost Classifier (Binary classification only)
    """

    def __init__(self, max_iterations=10, estimator=WeightedDecisionStump, estimator_params={}):
        self.max_iterations = max_iterations
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):

        # init weights
        w = np.full(len(y), 1/len(y))

        for _ in range(self.max_iterations):
            clf = self.estimator(**self.estimator_params)
            y_pred = clf.fit(X, y, w).predict(X)

            epsilon = 1.0 * np.sum((y != y_pred).astype(int) * w) / np.sum(w)
            alpha = np.log( (1.0 - epsilon) / epsilon )

            # update weights
            w = w * np.exp(alpha * (y != y_pred).astype(int))

            # store m-th model & alpha
            self.estimators.append(clf)
            self.alphas.append(alpha)

        return self

    def predict(self, X):

        # y_preds.shape is (max_iterations, len(X))
        y_preds = np.array([clf.predict(X) for clf in self.estimators])
        
        # replace 0 with -1 so the label is not {1,0} but {1,-1}
        y_preds[y_preds == 0] = -1
        
        # weighted majority vote
        y_pred = np.sign(np.sum(np.array(self.alphas)[:,None] * y_preds, axis=0))

        # back to {1,0} labels
        y_pred[y_pred == -1] = 0

        return y_pred








