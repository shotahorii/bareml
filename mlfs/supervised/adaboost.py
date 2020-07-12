"""
Ada Boost

References:
[1] Y. Freund and R.E. Schapire (1997). 
    A decision-theoretic generalization of on-line learning and an application to boosting. 
    Journal of Computer and System Sciences. 55: 119–139.
    (https://www.sciencedirect.com/science/article/pii/S002200009791504X)
[2] Y. Freund and R.E. Schapire (1996). Experiments with a New Boosting Algorithm
    Machine Learning: Proceedings of the Thirteenth International Conference, 1996.
    (https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf)
[3] Y. Freund and R.E. Schapire (1999). A Short Introduction to Boosting
    Journal of Japanese Society for Artificial Intelligence. 14(5):771-780.
    (http://www.site.uottawa.ca/~stan/csi5387/boost-tut-ppr.pdf)
[4] R. Schapire and Y. Singer (1999).
    Improved boosting algorithms using confidence-rated prediction. 
    Machine Learning 37(3). 297–336.
    (https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1999-ML-Improved%20boosting%20algorithms%20using%20confidence-rated%20predictions%20(Schapire%20y%20Singer).pdf)

[5] C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 657-663.
[6] Y. Hirai (2012). はじめてのパターン認識. 森北出版. 188-192.
[7] Nikolaos Nikolaou. Introduction to AdaBoost. (https://nnikolaou.github.io/files/Introduction_to_AdaBoost.pdf)
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from mlfs.utils.transformers import prob2binary
from mlfs.supervised.base_classes import Classifier, Regressor
from mlfs.supervised.decision_trees import WeightedDecisionStump

class AdaBoost(Classifier):
    """
    AdaBoost Classifier (Binary classification only)
    There're many different variations exist. 
    This implementation is based on the refference [3].
    !!!!! Change the implementation to align with [3] !!!!!
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

            # calculate weighted error
            epsilon = np.sum((y != y_pred).astype(int) * w) / np.sum(w)

            # update weights
            alpha = 0.5 * np.log((1.0 - epsilon) / epsilon)
            w = w * np.exp(alpha * np.sign((y != y_pred).astype(int) - 0.5))
            w = w/np.sum(w)

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


class AdaBoostM1(Classifier):
    """
    AdaBoost.M1 Classifier
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

            # calculate weighted error
            epsilon = 1.0 * np.sum((y != y_pred).astype(int) * w) / np.sum(w)

            # if error rate is equal to or greater than 1/2, stop iteration.
            if epsilon >= 0.5:
                break

            # update weights
            alpha = np.log( (1.0 - epsilon) / epsilon )
            w = w * np.exp(alpha * (y != y_pred).astype(int))

            # store m-th model & alpha
            self.estimators.append(clf)
            self.alphas.append(alpha)

        return self





