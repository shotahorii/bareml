"""
Stacking

Author: Shota Horii <sh.sinker@gmail.com>

References:
"""

import math
import numpy as np
from abc import ABC, abstractmethod

from ..base import Classifier, Regressor, Ensemble
from ..utils.validators import cross_val_predict

class Stacking(Ensemble):
    """
    
    Parameters
    ----------
    estimators: list of instances of classifier or regressor

    final_estimator: an instance of classifier or regressor
    """
    
    def __init__(self, estimators, final_estimator, cv=5):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv

        if isinstance(self, Classifier):
            self.stratified = True
        elif isinstance(self, Regressor):
            self.stratified = False 
        else:
            raise ValueError('This needs to be instanciate as Classifier or Regressor.')

    def _fit(self, X, y):

        X_final = None

        # make training data set for the final estimator
        # by cross_val_predict with each base estimator.
        # Then train the base estimators with entire X, y.
        for estimator in self.estimators:

            pred = cross_val_predict(estimator, X, y, stratified=self.stratified)

            # if pred is 1d array, convert it as a column vector
            if pred.ndim == 1:
                pred = pred[:,None]
            # if pred is 2d array (i.e. multi class classification),
            # remove 1 column as the last column doesn't add any info.
            else:
                pred = np.delete(pred, -1, 1)
            
            # merge the prediction from this estimator into the input 
            # data of the final estimator
            if X_final is None:
                X_final = pred
            else:
                X_final = np.concatenate([X_final, pred],axis=1)

            # Finally ,train the base estimator with entire X, y
            estimator.fit(X, y)

        # train final estimator
        self.final_estimator.fit(X_final, y)

        return self

    def _predict(self, X):

        X_final = None

        for estimator in self.estimators:

            pred = estimator.predict(X)

            # if pred is 1d array, convert it as a column vector
            if pred.ndim == 1:
                pred = pred[:,None]
            # if pred is 2d array (i.e. multi class classification),
            # remove 1 column as the last column doesn't add any info.
            else:
                pred = np.delete(pred, -1, 1)
            
            # merge the prediction from this estimator into the input 
            # data of the final estimator
            if X_final is None:
                X_final = pred
            else:
                X_final = np.concatenate([X_final, pred],axis=1)

        return self.final_estimator.predict(X_final)


class StackingClassifier(Stacking, Classifier):

    def __init__(self, estimators, final_estimator, cv=5):
        super().__init__(estimators, final_estimator, cv=5)

    
class StackingRegressor(Stacking, Regressor):

    def __init__(self, estimators, final_estimator, cv=5):
        super().__init__(estimators, final_estimator, cv=5)