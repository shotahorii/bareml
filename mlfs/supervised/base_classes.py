"""
base classes
"""

# Author: Shota Horii <sh.sinker@gmail.com>

from abc import ABC, abstractmethod
from mlfs.utils.metrics import accuracy, precision_recall_f1, mae, rmse, r_squqred

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
        y_pred = self.predict(X)

        rmse_ = rmse(y, y_pred)
        mae_ = mae(y, y_pred)
        r2 = r_squqred(y, y_pred)

        return {'rmse':rmse_, 'mae':mae_, 'r_squared':r2}


