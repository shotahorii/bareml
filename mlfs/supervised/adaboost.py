"""
AdaBoost

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
[5] T. Hastie, S. Rosset, J. Zhu, H. Zou (2009). Multi-class AdaBoost.
    Statistics and its Interface, Volume 2 (2009). 349-360.
    (https://www.intlpress.com/site/pub/files/_fulltext/journals/sii/2009/0002/0003/SII-2009-0002-0003-a008.pdf)
[6] D.P. Solomatine, D.L. Shrestha (2004). 
    AdaBoost.RT: a boosting algorithm for regression problems.
    2004 IEEE International Joint Conference on Neural Networks (IEEE Cat. No.04CH37541).
    (https://www.researchgate.net/publication/4116773_AdaBoostRT_A_boosting_algorithm_for_regression_problems)
[7] H. Drucker (1997). Improving Regressors using Boosting Techniques.
    ICML '97: Proceedings of the Fourteenth International Conference on Machine LearningJuly 1997. 107–115.
    (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.314&rep=rep1&type=pdf)
    
[8] C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 657-663.
[9] Y. Hirai (2012). はじめてのパターン認識. 森北出版. 188-192.
[10] Nikolaos Nikolaou. Introduction to AdaBoost. (https://nnikolaou.github.io/files/Introduction_to_AdaBoost.pdf)
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from mlfs.utils.transformers import prob2binary, binary2sign, sign2binary, BinaryOnehotEncoder
from mlfs.supervised.base_classes import Classifier, Regressor
from mlfs.supervised.decision_trees import WeightedDecisionStump

class AdaBoost(Classifier):
    """
    AdaBoost Classifier (Binary classification only)
    There're many different variations exist. 
    This implementation is based on the refference [3].
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
            
            # assign 1 to the samples which are wrongly predicted, 
            # and assign 0 to the samples which are correctly predicted.
            y_error = (y != y_pred).astype(int)

            # calculate weighted error
            epsilon = np.sum(y_error * w) / np.sum(w)

            # avoid 0 division
            epsilon = np.clip(epsilon, 1e-15, 1 - 1e-15)

            # calculate alpha: how good this prediction is.
            # when ipsilon (weighted error) > 1/2, alpha < 0. 
            alpha = 0.5 * np.log((1.0 - epsilon) / epsilon)

            # update weights
            # Note that: 
            #   We don't have to flip the prediction even if epsilon > 1/2 
            #   because of below w update logic. 
            #   if we flip the prediction, sign of both alpha and binary2sign(y_error) flip.
            #   as the result, alpha * binary2sign(y_error) doesn't change. 
            # Also note:
            #   Minus alpha works fine in prediction as well, as long as we use {-1,1} for label. 
            w = w * np.exp(alpha * binary2sign(y_error))
            w = w/np.sum(w) # normalisation

            # store m-th model & alpha
            self.estimators.append(clf)
            self.alphas.append(alpha)

        return self

    def predict(self, X):

        # y_preds.shape is (max_iterations, len(X))
        y_preds = np.array([clf.predict(X) for clf in self.estimators])
        
        # convert labels from {0,1} to {-1,1}
        y_preds = binary2sign(y_preds)
        
        # weighted majority vote
        y_pred = np.sign(np.sum(np.array(self.alphas)[:,None] * y_preds, axis=0))

        # back to {1,0} labels
        y_pred = sign2binary(y_pred)

        return y_pred


class AdaBoostM1(Classifier):
    """
    AdaBoost.M1 Classifier
    Implementation based on the refference [2].
    """

    def __init__(self, max_iterations=10, estimator=WeightedDecisionStump, estimator_params={}):
        self.max_iterations = max_iterations
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.estimators = []
        self.betas = []
        self.onehot = BinaryOnehotEncoder()

    def fit(self, X, y):

        # init weights
        w = np.full(len(y), 1/len(y))

        # treat binary classification as same data format as multi-class
        if y.ndim == 1:
            y_onehot = self.onehot.encode(y)
        else:
            y_onehot = y

        for _ in range(self.max_iterations):
            clf = self.estimator(**self.estimator_params)
            y_pred = clf.fit(X, y, w).predict(X) # fit with original y NOT y_onehot

            # treat binary classification as same data format as multi-class
            if y.ndim == 1:
                y_pred = self.onehot.encode(y_pred)
            
            # assign 1 to the samples which are wrongly predicted, 
            # and assign 0 to the samples which are correctly predicted.
            y_error = (~(y_onehot == y_pred).all(axis=1)).astype(int)

            # calculate weighted error
            epsilon = np.sum(y_error * w) / np.sum(w)

            # avoid 0 division
            epsilon = np.clip(epsilon, 1e-15, 1 - 1e-15)
            
            # if weighted error is bigger than 1/2, terminate the training.
            if epsilon > 0.5:
                break

            # calculate beta: how bad this prediction is.
            # when ipsilon (weighted error) = 1/2, beta = 1. 
            # since epsilon <= 0.5 due to break condition above, beta in [0,1]
            beta = epsilon / (1.0 - epsilon)

            # assign 0 to the samples which are wrongly predicted, 
            # and assign 1 to the samples which are correctly predicted.
            y_correct = (y_onehot == y_pred).all(axis=1).astype(int)

            # update weights
            # w * beta for samples correctly predicted, w * 1 for samples wrongly predicted.
            w = w * np.power(beta, y_correct)
            w = w/np.sum(w) # normalisation

            # store m-th model & beta
            self.estimators.append(clf)
            self.betas.append(beta)

        return self

    def predict(self, X):

        # y_preds.shape is (max_iterations, len(X), c) where c = number of classes
        y_preds = np.array([clf.predict(X) for clf in self.estimators])

        # treat binary classification as same data format as multi-class
        if y_preds[0].ndim == 1:
            y_preds = np.array([self.onehot.encode(y_pred) for y_pred in y_preds])
        
        # weighted majority vote
        y_pred = np.zeros(y_preds[0].shape)
        for t in range(len(y_preds)): # number of estimators
            y_pred = y_pred + y_preds[t] * np.log(1.0/beta[t])
        
        y_pred = prob2binary(y_pred)

        return self.onehot.decode(y_pred)

    
class AdaBoostSamme(Classifier):
    """
    AdaBoost SAMME Classifier
    Implementation based on the refference [5].
    """

    def __init__(self, max_iterations=10, estimator=WeightedDecisionStump, estimator_params={}):
        self.max_iterations = max_iterations
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.estimators = []
        self.alphas = []
        self.onehot = BinaryOnehotEncoder()

    def fit(self, X, y):

        # init weights
        w = np.full(len(y), 1/len(y))

        # treat binary classification as same data format as multi-class
        if y.ndim == 1:
            y_onehot = self.onehot.encode(y)
        else:
            y_onehot = y

        k = y_onehot.shape[1]

        for _ in range(self.max_iterations):
            clf = self.estimator(**self.estimator_params)
            y_pred = clf.fit(X, y, w).predict(X) # fit with original y NOT y_onehot

            # treat binary classification as same data format as multi-class
            if y.ndim == 1:
                y_pred = self.onehot.encode(y_pred)
            
            # assign 1 to the samples which are wrongly predicted, 
            # and assign 0 to the samples which are correctly predicted.
            y_error = (~(y_onehot == y_pred).all(axis=1)).astype(int)

            # calculate weighted error
            epsilon = np.sum(y_error * w) / np.sum(w)

            # avoid 0 division
            epsilon = np.clip(epsilon, 1e-15, 1 - 1e-15)

            # calculate alpha: how good this prediction is.
            # alpha > 0 when 1 - epsilon > 1/k
            alpha = np.log((1.0 - epsilon) / epsilon) + np.log(k-1)

            # update weights
            w = w * np.exp(alpha * y_error)
            w = w/np.sum(w) # normalisation

            # store m-th model & beta
            self.estimators.append(clf)
            self.alphas.append(alpha)

        return self

    def predict(self, X):

        # y_preds.shape is (max_iterations, len(X), c) where c = number of classes
        y_preds = np.array([clf.predict(X) for clf in self.estimators])

        # treat binary classification as same data format as multi-class
        if y_preds[0].ndim == 1:
            y_preds = np.array([self.onehot.encode(y_pred) for y_pred in y_preds])
        
        # weighted majority vote
        y_pred = np.zeros(y_preds[0].shape)
        for t in range(len(y_preds)): # number of estimators
            y_pred = y_pred + alphas[t] * y_preds[t]
        
        y_pred = prob2binary(y_pred)

        return self.onehot.decode(y_pred)



