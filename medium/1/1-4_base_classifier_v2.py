from abc import ABCMeta, abstractmethod
import numpy as np

class Classifier(metaclass=ABCMeta):
    """
    Base class for all classifier implementations. 
    """

    def _validate_X(X):
        """ Validates input X """
        X = np.array(X) # if X is python native list, convert it to np.ndarray
        assert np.issubdtype(X.dtype, np.number), 'Data type of X needs to be numeric.'

        return X

    def _validate_y(self, y):
        y = np.array(y) # if y is python native list, convert it to np.ndarray
        assert y.ndim == 2, 'y needs to be a 2d array.'
        assert np.issubdtype(y.dtype, np.number), 'Data type of y needs to be numeric.'
        assert np.isin(y, [0,1]).all(), 'Element in y needs to be 0 or 1.'
        assert (y.sum(axis=1)==1).all(), 'Each sample needs to be asigned to exactly 1 class.'
        
        return y

    def fit(self, X, y):
        """ Train the model. """
        X = self._validate_X(X)
        y = self._validate_y(y)
        assert len(X) == len(y), 'X and y need to have the same number of elements.'
        
        return self._fit(X, y)

    def predict(self, X):
        """ Predict. """
        X = self._validate_X(X)
        return self._predict(X)

    @abstractmethod
    def _fit(self, X, y):
        """ Actual implementation of fit """
        pass

    @abstractmethod
    def _predict(self, X):
        """ Actual implementation of predict """
        pass

    def score(self, X, y):
        """ Calculate accuracy. """
        y_pred = self.predict(X)
        accuracy = some_function_to_calc_accuracy(y, y_pred)
        return accuracy