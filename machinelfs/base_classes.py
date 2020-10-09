"""
base classes
"""

# Author: Shota Horii <sh.sinker@gmail.com>

from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np

from machinelfs.utils.metrics import accuracy, precision_recall_f1, mae, rmse, r_squqred


class Estimator(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y):
        """ Train the model. """
        pass
    
    @abstractmethod
    def predict(self, X):
        """ Returns predicted value for input samples. """
        pass

    @abstractmethod
    def score(self, X, y):
        """ Returns evaluation metrics. """
        pass

    def _validate_y(self, y):
        """ Validates input y for training. """
        return y

    def _validate_X(self, X):
        """ Validates input X for training. """
        X = np.array(X)

        if X.dtype not in ['int64','float64','uint8']:
            raise ValueError('Data type of X needs to be int or float.')
        elif X.ndim != 1 and X.ndim != 2:
            raise ValueError('X needs to be a 1d array or 2d array.')
        elif np.isnan(X).any():
            raise ValueError('There is at least 1 null element in X.')
        elif np.isinf(X).any():
            raise ValueError('There is at least 1 inf/-inf element in X.')

        if X.ndim == 1:
            # convert to column vector 
            # e.g. [1,2,3] -> [[1],[2],[3]]
            X = np.expand_dims(X, axis=1)
            #X = X[:,None] 

        return X

    def _validate_w(self, w):
        """ Validates input weight for training. """
        if w is None:
            return w
            
        w = np.array(w)

        if w.dtype not in ['int64','float64','uint8']:
            raise ValueError('Data type of w needs to be int or float.')
        elif w.ndim != 1:
            raise ValueError('w needs to be a 1d array.')
        elif np.isnan(w).any():
            raise ValueError('There is at least 1 null element in w.')
        elif np.isinf(w).any():
            raise ValueError('There is at least 1 inf/-inf element in w.')
        elif (w < 0).any():
            raise ValueError('w cannot be minus.')

        return w

    def _validate_Xy(self, X, y):
        """
        Validates input X and y for training. 
        Every fit() function must call this method 
        at the first line to validate input X and y.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)

        if y is not None and len(X) != len(y):
            raise ValueError('Length of X and y need to be same.')

        return X, y

    def _validate_Xyw(self, X, y, w):
        """
        Validates input X and y for training. 
        Every fit() function must call this method 
        at the first line to validate input X and y.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)
        w = self._validate_w(w)

        if y is not None and len(X) != len(y):
            raise ValueError('Length of X and y need to be same.')

        if w is not None and len(X) != len(w):
            raise ValueError('Length of X and w need to be same.')

        return X, y, w


class Regressor(Estimator):

    def score(self, X, y):
        """ Returns various evaluation metrics. """
        y_pred = self.predict(X)

        rmse_ = rmse(y, y_pred)
        mae_ = mae(y, y_pred)
        r2 = r_squqred(y, y_pred)

        return {'rmse':rmse_, 'mae':mae_, 'r_squared':r2}

    def _validate_y(self, y):
        y = np.array(y)

        if y.dtype not in ['int64','float64','uint8']:
            raise ValueError('Data type of y needs to be int or float.')
        elif y.ndim != 1:
            raise ValueError('y needs to be a 1d array.')
        elif np.isnan(y).any():
            raise ValueError('There is at least 1 null element in y.')
        elif np.isinf(y).any():
            raise ValueError('There is at least 1 inf/-inf element in y.')

        return y


class Classifier(Estimator):
    
    def predict_proba(self, X):
        """
        Returns estimated probability for each class for each sample. 
        This can be left without overridden, if the classifier isn't 
        eligible to provide probability estimates. 
        """
        raise NotImplementedError('This classifier is not eligible to provide probability.')

    def score(self, X, y):
        """ Returns various evaluation metrics. """
        y_pred = self.predict(X)
        
        accuracy_ = accuracy(y, y_pred)
        precision_, recall_, f1_ = precision_recall_f1(y, y_pred)

        return {'accuracy':accuracy_, 'precision':precision_, 'recall':recall_, 'f1':f1_}

    def _validate_y(self, y):
        y = np.array(y)

        if y.dtype not in ['int64','float64','uint8']:
            raise ValueError('Data type of y needs to be int or float.')
        elif y.ndim != 1 and y.ndim != 2:
            raise ValueError('y needs to be a 1d array or 2d array.')
        elif not np.isin(y, [0,1]).all():
            raise ValueError('Element in y needs to be 0 or 1.')
        elif y.ndim == 2 and not (y.sum(axis=1)==1).all():
            raise ValueError('Each sample needs to be asigned to exactly 1 class.')
        
        return y


class BinaryClassifier(Classifier):
    
    def _validate_y(self, y):
        y = np.array(y)

        if y.dtype not in ['int64','float64','uint8']:
            raise ValueError('Data type of y needs to be int or float.')
        elif y.ndim != 1:
            raise ValueError('y needs to be a 1d array.')
        elif not np.isin(y, [0,1]).all():
            raise ValueError('element in y needs to be 0 or 1.')
        
        return y


class Ensemble(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, estimators=[], base_estimator=None):
        self.estimators = estimators
        self.base_estimator = base_estimator

    def _make_estimator(self, append=True):
        estimator = deepcopy(self.base_estimator)
        
        if append:
            self.estimators.append(estimator)
        
        return estimator