"""
Naive Bayes Classifier

Author: Shota Horii <sh.sinker@gmail.com>
Test: tests/test_naive_bayes.py

References:

ToDo: 
- multinomial NB
- smoothing
"""

import math
import numpy as np

from ..base import Classifier
from ..utils.probability_distribution import Bernoulli, Binomial, Poisson, Gaussian
from ..utils.manipulators import prob2binary, real2binary


class NaiveBayes(Classifier):

    def __init__(self, prob):
        self.prob = prob
        self.priors = None
        self.params = None
    
    def _fit(self, X, y):

        n_classes = y.shape[1]
        n_features = X.shape[1]

        # init params
        params = {}
        for p in self.prob.param_names:
            params[p] = np.zeros((n_classes, n_features))
        
        # init prior probs
        priors = np.zeros(n_classes)

        for i in range(n_classes):
            # update log prior prob of i-th class
            priors[i] = np.log( y[:,i].sum()/len(y) )

            # X of data in the i-th class
            X_i = X[ y[:,i]==1 ]

            for j in range(n_features):
                # array of j-th feature of the data in i-th class
                X_i_j = X_i[:,j]

                for p in self.prob.param_names:
                    params[p][i, j] = self.prob.mle(X_i_j, param=p)

        self.priors = priors
        self.params = params

        return self

    def _predict(self, X):

        n_samples, n_features = X.shape
        n_classes = len(self.priors)
        
        # init posterior as prior 
        posteriors = np.tile(self.priors,(n_samples,1)).T

        for i in range(n_classes):
            for j in range(n_features):
                X_j = X[:,j]
                params = {}
                for p in self.prob.param_names:
                    params[p] = self.params[p][i,j]

                posteriors[i] += self.prob.llh(X_j, **params)

        posteriors = posteriors.T

        # probability to {0,1}
        y_pred = prob2binary(posteriors)
        return y_pred
                

class GaussianNB(NaiveBayes):
    def __init__(self):
        super().__init__(Gaussian())


class BernoulliNB(NaiveBayes):
    def __init__(self, binarise=0.0):
        self.binarise = binarise
        super().__init__(Bernoulli())

    def _fit(self, X, y):
        # binarise the input features 
        # e.g. [0, 1, 2, 0.5] -> [0, 1, 1, 1] when self.binarise = 0.0
        X = real2binary(X, threshold=self.binarise, inclusive=False)
        return super()._fit(X, y)

    def _predict(self, X):
        # binarise the input features 
        # e.g. [0, 1, 2, 0.5] -> [0, 1, 1, 1] when self.binarise = 0.0
        X = real2binary(X, threshold=self.binarise, inclusive=False)
        return super()._predict(X)