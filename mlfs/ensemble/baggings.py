"""
Bagging

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import random
import numpy as np

from mlfs.base_classes import Classifier, Regressor, Ensemble
from mlfs.utils.transformers import prob2binary
from mlfs.utils.sampling import bootstrap_sampling
from mlfs.supervised.decision_trees import RandomTreeClassifier, RandomTreeRegressor

class Bagging(Ensemble):
    
    def __init__(self, estimator, n_estimators=50, sampling_ratio=1.0):
        super().__init__(base_estimator=estimator)
        self.n_estimators = n_estimators
        self.sampling_ratio = sampling_ratio

    def fit(self, X, y):

        for _ in range(self.n_estimators):

            X_bootstrap, y_bootstrap = bootstrap_sampling(X, y, self.sampling_ratio)
            estimator = self._make_estimator()
            estimator.fit(X_bootstrap, y_bootstrap)
        
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

        base_estimator = RandomTreeClassifier(
                    criterion=criterion,
                    max_features=max_features,
                    max_depth=max_depth,
                    min_impurity_decrease=min_impurity_decrease
                    )      

        super().__init__(
            estimator=base_estimator,
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

        base_estimator = RandomTreeRegressor(
                    criterion=criterion,
                    max_features=max_features,
                    max_depth=max_depth,
                    min_impurity_decrease=min_impurity_decrease
                    )        

        super().__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            sampling_ratio=sampling_ratio
        )
    
    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

            

