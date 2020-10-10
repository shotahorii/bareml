"""
Naive Bayes Classifier

Author: Shota Horii <sh.sinker@gmail.com>
Test: 

References:

ToDo: 
- bernoulli & multinomial NB
- smoothing
"""

import math
import numpy as np

from bareml import Classifier
from bareml.utils.probability_distribution import Bernoulli, Binomial, Poisson, Gaussian
from bareml.utils.manipulators import prob2binary, OnehotEncoder


class NaiveBayes(Classifier):

    def __init__(self, prob):
        self.prob = prob
        self.priors = None
        self.params = None
        self.onehot = OnehotEncoder()
    
    def fit(self, X, y):

        # validate the input data
        X, y = self._validate_Xy(X, y)
        
        # if binary classification, change the format of y
        # e.g. [0,1,1,0] -> [[1,0],[0,1],[0,1],[1,0]]
        if y.ndim == 1:
            y = self.onehot.encode(y)

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

    def predict(self, X):
        # validate the input data
        X = self._validate_X(X)

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

        if y_pred.shape[1]==2: # binary classification
            # e.g. [[1,0],[0,1],[0,1],[1,0]] -> [0,1,1,0]
            return self.onehot.decode(y_pred)
        else:
            return y_pred
                

class GaussianNB(NaiveBayes):
    def __init__(self):
        super().__init__(Gaussian())

class BernoulliNB(NaiveBayes):
    pass