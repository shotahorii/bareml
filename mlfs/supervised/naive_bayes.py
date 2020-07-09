"""
Naive Bayes Classifier

References:

ToDo: 
bernoulli & multinomial NB
smoothing
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
from mlfs.utils.probability_distribution import Bernoulli, Binomial, Poisson, Gaussian
from mlfs.supervised.base_classes import Classifier
from mlfs.utils.misc import prob2binary

class NaiveBayes(Classifier):

    def __init__(self, prob):
        self.prob = prob
        self.priors = None
        self.params = None
    
    def fit(self, X, y):
        
        # if binary classification, change the format of y
        # to make it same as multi class classification
        if y.ndim == 1:
            y_inv = (y!=1).astype(int)
            y = np.append([y], [y_inv], axis=0).T

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

        # probability to 1/0
        y_pred = prob2binary(posteriors)

        return y_pred
                

class GaussianNB(NaiveBayes):
    def __init__(self):
        super().__init__(Gaussian())

class BernoulliNB(NaiveBayes):
    pass