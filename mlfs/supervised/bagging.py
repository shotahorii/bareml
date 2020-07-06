"""
Bagging

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import random
import numpy as np
from mlfs.utils.misc import prob2binary
from mlfs.utils.metrics import entropy, gini_impurity, variance, mean_deviation
from mlfs.supervised.base_classes import Classifier
from mlfs.supervised.decision_trees import DecisionTreeClassifier, DecisionTreeRegressor

class Bagging:
    
    def __init__(self, estimator, estimator_params={}, n_estimators=5, sampling_ratio=1.0):
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.n_estimators = n_estimators
        self.sampling_ratio = sampling_ratio
        self.estimators = []

    def fit(self, X, y):

        n_samples = int( round( len(X) * self.sampling_ratio ) )

        for _ in range(self.n_estimators):
            sample_idx = random.choices( np.arrange(len(X)), k=n_samples)
            estimator = self.estimator(**self.estimator_params)
            estimator.fit(X[sample_idx], y[sample_idx])
            self.estimators.append(estimator)
        
        return self

    def predict(self, X):

        preds = [ estimator.predict(X) for estimator in self.estimators ]
        
        y_pred = np.mean(preds, axis=0)

        if isinstance(self.estimator, Classifier):
            y_pred = prob2binary(y_pred)
        
        return y_pred

        
            

            

