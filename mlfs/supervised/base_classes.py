"""
base classes
"""

# Author: Shota Horii <sh.sinker@gmail.com>

from abc import ABC, abstractmethod
from mlfs.utils.metrics import accuracy, precision_recall_f1

class Classifier(ABC):
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        y_pred = self.predict(X)
        
        accuracy_ = accuracy(y, y_pred)
        precision_, recall_, f1_ = precision_recall_f1(y, y_pred)

        return {'accuracy':accuracy_, 'precision':precision_, 'recall':recall_, 'f1':f1_}


class Regressor(ABC):
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        pass

class Weighted(ABC):
    """ 
    Weighted samples are used for training.
    fit() takes X, y and weights as parameters.
    """
    pass