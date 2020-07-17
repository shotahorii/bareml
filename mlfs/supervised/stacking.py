"""
Stacking

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from mlfs.supervised.base_classes import Classifier, Regressor
from mlfs.utils.validation import KFold

class Stacking:
    """
    
    Parameters
    ----------
    estimators: list of instances of classifier or regressor

    final_estimator: an instance of classifier or regressor
    """
    
    def __init__(self, estimators, final_estimator, cv=5):
        self.estimators = estimators 
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        
        kf = KFold(n_splits=self.cv)

        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
