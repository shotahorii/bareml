"""
Bagging

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import random
import numpy as np
from mlfs.utils.misc import prob2binary
from mlfs.supervised.base_classes import Classifier, Regressor
from mlfs.supervised.decision_trees import RandomTreeClassifier, RandomTreeRegressor

class Bagging:
    
    def __init__(self, estimator, estimator_params={}, n_estimators=50, sampling_ratio=1.0):
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.n_estimators = n_estimators
        self.sampling_ratio = sampling_ratio
        self.estimators = []

    def fit(self, X, y):

        n_samples = int( round( len(X) * self.sampling_ratio ) )

        for _ in range(self.n_estimators):
            sample_idx = random.choices( np.arange(len(X)), k=n_samples)
            estimator = self.estimator(**self.estimator_params)
            estimator.fit(X[sample_idx], y[sample_idx])
            self.estimators.append(estimator)
        
        return self

    def predict(self, X):

        preds = [ estimator.predict(X) for estimator in self.estimators ]
        
        y_pred = np.mean(preds, axis=0)

        if isinstance(self.estimators[0], Classifier):
            y_pred = prob2binary(y_pred)
        
        return y_pred


class RandomForestClassifier(Bagging, Classifier):

    def __init__(self, 
                criterion='gini',
                max_features='sqrt',
                max_depth=None, 
                min_impurity_decrease=None,
                n_estimators=50, 
                sampling_ratio=1.0):

        estimator_params = {
            'criterion': criterion,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_impurity_decrease': min_impurity_decrease
        }        

        super().__init__(
            estimator=RandomTreeClassifier,
            estimator_params=estimator_params,
            n_estimators=n_estimators,
            sampling_ratio=sampling_ratio
        )

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

class RandomForestRegressor(Bagging, Regressor):

    def __init__(self, 
                criterion='mse',
                max_features='sqrt',
                max_depth=None, 
                min_impurity_decrease=None,
                n_estimators=50, 
                sampling_ratio=1.0):

        estimator_params = {
            'criterion': criterion,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_impurity_decrease': min_impurity_decrease
        }        

        super().__init__(
            estimator=RandomTreeRegressor,
            estimator_params=estimator_params,
            n_estimators=n_estimators,
            sampling_ratio=sampling_ratio
        )
    
    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

            

