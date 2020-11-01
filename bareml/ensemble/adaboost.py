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
[7] D.P. Solomatine, D.L. Shrestha (2006). 
    Experiments with AdaBoost.RT, an Improved Boosting Scheme for Regression.
    Neural Computation 18(7):1678-1710.
    (https://www.researchgate.net/publication/220499818_Experiments_with_AdaBoostRT_an_Improved_Boosting_Scheme_for_Regression)
[8] H. Drucker (1997). Improving Regressors using Boosting Techniques.
    ICML '97: Proceedings of the Fourteenth International Conference on Machine LearningJuly 1997. 107–115.
    (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.314&rep=rep1&type=pdf)
    
[9] C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 657-663.
[10] Y. Hirai (2012). はじめてのパターン認識. 森北出版. 188-192.
[11] Nikolaos Nikolaou. Introduction to AdaBoost. (https://nnikolaou.github.io/files/Introduction_to_AdaBoost.pdf)
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from ..base import Classifier, Regressor, BinaryClassifier, Ensemble
from ..utils.manipulators import prob2binary, binary2sign, sign2binary, OnehotEncoder
from ..supervised.decision_trees import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils.metrics import absolute_relative_errors, absolute_errors

class AdaBoost(Ensemble, BinaryClassifier):
    """
    AdaBoost Classifier (Binary classification only)
    There're many different variations exist. 
    This implementation is based on the refference [3].
    """

    def __init__(self, max_iter=10, estimator=DecisionTreeClassifier(max_depth=1)):
        super().__init__(base_estimator=estimator)
        self.max_iter = max_iter
        self.alphas = []

    def _fit(self, X, y):

        # init weights
        w = np.full(len(y), 1/len(y))

        for _ in range(self.max_iter):
            clf = self._make_estimator()
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

            # store m-th alpha
            self.alphas.append(alpha)

        return self

    def _predict(self, X):

        # y_preds.shape is (max_iterations, len(X))
        y_preds = np.array([clf.predict(X) for clf in self.estimators])
        
        # convert labels from {0,1} to {-1,1}
        # y_preds = binary2sign(y_preds)
        
        # weighted majority vote
        y_pred = np.sign(np.sum(np.array(self.alphas)[:,None] * y_preds, axis=0))

        # back to {1,0} labels
        # y_pred = sign2binary(y_pred)

        return y_pred


class AdaBoostM1(Ensemble, Classifier):
    """
    AdaBoost.M1 Classifier
    Implementation based on the refference [2].
    """

    def __init__(self, max_iter=10, estimator=DecisionTreeClassifier(max_depth=1)):
        super().__init__(base_estimator=estimator)
        self.max_iter = max_iter
        self.betas = []
        self.onehot = OnehotEncoder()

    def _fit(self, X, y):

        # init weights
        w = np.full(len(y), 1/len(y))

        # treat binary classification as same data format as multi-class
        if y.ndim == 1:
            y_onehot = self.onehot.fit_transform(y)
        else:
            y_onehot = y

        for _ in range(self.max_iter):
            clf = self._make_estimator()
            y_pred = clf.fit(X, y, w).predict(X) # fit with original y NOT y_onehot

            # treat binary classification as same data format as multi-class
            if y.ndim == 1:
                y_pred = self.onehot.transform(y_pred)
            
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

            # store m-th beta
            self.betas.append(beta)

        return self

    def _predict(self, X):

        # y_preds.shape is (max_iterations, len(X), c) where c = number of classes
        y_preds = np.array([clf.predict(X) for clf in self.estimators])

        # treat binary classification as same data format as multi-class
        if y_preds[0].ndim == 1:
            y_preds = np.array([self.onehot.transform(y_pred) for y_pred in y_preds])
        
        # weighted majority vote
        y_pred = np.zeros(y_preds[0].shape)
        for t in range(len(y_preds)): # number of estimators
            y_pred = y_pred + y_preds[t] * np.log(1.0/self.betas[t])
        
        y_pred = prob2binary(y_pred)

        return self.onehot.inverse_transform(y_pred)

    
class AdaBoostSamme(Ensemble, Classifier):
    """
    AdaBoost SAMME Classifier
    Implementation based on the refference [5].
    """

    def __init__(self, max_iter=10, estimator=DecisionTreeClassifier(max_depth=1)):
        super().__init__(base_estimator=estimator)
        self.max_iter = max_iter
        self.alphas = []
        self.onehot = OnehotEncoder()

    def _fit(self, X, y):

        # init weights
        w = np.full(len(y), 1/len(y))

        # treat binary classification as same data format as multi-class
        if y.ndim == 1:
            y_onehot = self.onehot.fit_transform(y)
        else:
            y_onehot = y

        k = y_onehot.shape[1]

        for _ in range(self.max_iter):
            clf = self._make_estimator()
            y_pred = clf.fit(X, y, w).predict(X) # fit with original y NOT y_onehot

            # treat binary classification as same data format as multi-class
            if y.ndim == 1:
                y_pred = self.onehot.transform(y_pred)
            
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

            # store m-th alpha
            self.alphas.append(alpha)

        return self

    def _predict(self, X):

        # y_preds.shape is (max_iterations, len(X), c) where c = number of classes
        y_preds = np.array([clf.predict(X) for clf in self.estimators])

        # treat binary classification as same data format as multi-class
        if y_preds[0].ndim == 1:
            y_preds = np.array([self.onehot.transform(y_pred) for y_pred in y_preds])
        
        # weighted majority vote
        y_pred = np.zeros(y_preds[0].shape)
        for t in range(len(y_preds)): # number of estimators
            y_pred = y_pred + self.alphas[t] * y_preds[t]
        
        y_pred = prob2binary(y_pred)

        if y_preds[0].ndim == 1:
            return self.onehot.inverse_transform(y_pred)
        else:
            return y_pred


class AdaBoostRT(Ensemble, Regressor):
    """
    AdaBoost.RT Regressor 
    Implementation based on the refference [7].
    """

    def __init__(self, threshold=0.05, max_iter=10, estimator=DecisionTreeRegressor(max_depth=1)):
        super().__init__(base_estimator=estimator)
        self.threshold = threshold
        self.max_iter = max_iter
        self.betas = []

    def _fit(self, X, y):

        # init weights
        w = np.full(len(y), 1/len(y))

        for _ in range(self.max_iter):
            reg = self._make_estimator()
            y_pred = reg.fit(X, y, w).predict(X)

            # compute absolute relative error for each sample
            are = absolute_relative_errors(y, y_pred)

            # assign 1 to the samples where the absolute relative error > threshold
            # and assign 0 to the samples where the absolute relative error <= threshold
            y_error = (are > self.threshold).astype(int)

            # calculate weighted error
            epsilon = np.sum(y_error * w) / np.sum(w)

            # calculate alpha: how bad this prediction is.
            beta = epsilon ** 2

            # assign 0 to the samples where the absolute relative error > threshold
            # and assign 1 to the samples where the absolute relative error <= threshold
            y_correct = (are <= self.threshold).astype(int)

            # update weights
            # w * beta for samples correctly predicted, w * 1 for samples wrongly predicted.
            w = w * np.power(beta, y_correct)
            w = w/np.sum(w) # normalisation

            # store m-th beta
            self.betas.append(beta)

        return self

    def _predict(self, X):

        # y_preds.shape is (max_iter, len(X))
        y_preds = np.array([reg.predict(X) for reg in self.estimators])
        
        # weighted majority vote
        y_pred = np.sum(np.log(1.0/self.betas)[:,None] * y_pred, axis=0) / np.log(1.0/self.betas).sum()

        return y_pred


class AdaBoostR2(Ensemble, Regressor):
    """
    AdaBoost.R2 Regressor 
    Implementation based on the refference [8].
    """

    def __init__(loss='linear', max_iter=10, estimator=DecisionTreeRegressor(max_depth=1)):
        super().__init__(base_estimator=estimator)
        self.max_iter = max_iter
        self.betas = []

        if loss=='linear':
            self.loss = self._linear_loss
        elif loss=='square':
            self.loss = self._square_loss
        elif loss=='exponential':
            self.loss = self._exponential_loss
        else:
            raise ValueError('loss needs to be "linear", "square" or "exponential".')

    def _linear_loss(self, y, y_pred):
        abs_errs = absolute_errors(y, y_pred)
        D = np.max(abs_errs)
        return abs_errs / D

    def _square_loss(self, y, y_pred):
        abs_errs = absolute_errors(y, y_pred)
        D = np.max(abs_errs)
        return np.power(abs_errs, 2) / np.power(D, 2)

    def _exponential_loss(self, y, y_pred):
        abs_errs = absolute_errors(y, y_pred)
        D = np.max(abs_errs)
        return 1 - np.exp(-abs_errs/D)

    def _fit(self, X, y):

        # number of samples in the training data
        N = len(y)

        # init weights
        w = np.ones(N)

        for _ in range(self.max_iter):
            
            # probability that training sample i is in the training set
            p = w / np.sum(w)

            # pick N samples (with replacement) to form the training set
            train_idx = np.random.choice(np.arrange(N), size=N, replace=True, p=p)

            # form the training set
            X_train = X[train_idx]
            y_train = y[train_idx]

            # train the weak leaner with the formed training set
            reg = self._make_estimator()
            y_pred = reg.fit(X_train, y_train).predict(X)

            # calculate a loss for each training sample
            y_error = self.loss(y, y_pred) # this is normalised to [0,1]

            # calculate average loss
            epsilon = np.sum(y_error * p)

            # avoid 0 division
            epsilon = np.clip(epsilon, 1e-15, 1 - 1e-15)

            # terminate the loop if the average loss less than 1/2.
            if epsilon > 0.5:
                break

            # beta is a measure of confidence in the predictor. 
            # Low beat means high confidence in the prediction.
            beta = epsilon / (1 - epsilon)

            # update the weights
            w = w * np.power(beta, 1 - y_error)

            # store m-th beta
            self.betas.append(beta)

        return self

    def _predict(self, X):

        # y_preds.shape is (max_iterations, len(X))
        y_preds = np.array([reg.predict(X) for reg in self.estimators])

        # sort t estimators prediction by value for each sample. 
        # for example, if we have 3 estimators with 4 samples to predict, 
        # y_preds matrix is like [[11,22,33,44], [12,21,30,40], [13,20,31,42]]
        # here, we want to sort predictions by value for each sample.
        # for example, sort to [[11,20,30,40], [12,21,31,42], [13,22,33,44]]
        # here, instead, we do it in transposed matrix and return index not actual values
        # so, first trasnpose to [[11,12,13],[22,21,20],[33,30,31],[44,40,42]]
        # then return sorted index [[0,1,2],[2,1,0],[1,2,0],[1,2,0]]
        sorted_idx = np.argsort(y_preds.T, axis=1)

        # log(1/betas)
        log_inv_betas = np.log(1.0/self.betas)

        # true / false matrix
        is_over_median = log_inv_betas[sorted_idx].cumsum(axis=1) >= 0.5 * np.sum(log_inv_betas)

        # the first position with true is the index of median
        median_idx = is_over_median.argmax(axis=1)

        # back to original index
        median_estimators = sorted_idx[np.arrange(len(X)), median_idx]

        return y_preds.T[np.arange(len(X)), median_estimators]

