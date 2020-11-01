from abc import ABCMeta, abstractmethod

class Classifier(metaclass=ABCMeta):
    """
    Base class for all classifier implementations. 
    """

    @abstractmethod
    def fit(self, X, y):
        """ 
        Train the model.
        Actual implementation should be made in each class inherits this. 
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict. 
        Actual implementation should be made in each class inherits this. 
        """
        pass

    def score(self, X, y):
        """ Calculate accuracy. """
        y_pred = self.predict(X)
        accuracy = some_function_to_calc_accuracy(y, y_pred)
        return accuracy