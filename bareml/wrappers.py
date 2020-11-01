"""
Wrapper classes
Can be used like pipeline.
e.g. clf = Polynomial(2, Scaled(StandardScaler(), LogisticRegression()))
     -> Transform X by 2-degree polynomial, then apply standard scalling to X, 
        finally perform Logistic Regression.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

from abc import ABC, abstractmethod

from .base import Estimator
from .utils.manipulators import polynomial_features


class Preprocessed(Estimator, ABC):
    """ Base class for preprocessed estimators """

    @abstractmethod
    def __init__(self, estimator):
        self.estimator = estimator

    @abstractmethod
    def _preprocess_X_fit(self, X):
        pass

    @abstractmethod
    def _preprocess_X_predict(self, X):
        pass

    def fit(self, X, y):
        X = self._preprocess_X_fit(X)
        self.estimator = self.estimator.fit(X, y)
        return self

    def predict(self, X):
        X = self._preprocess_X_predict(X)
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        X = self._preprocess_X_predict(X)
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        X = self._preprocess_X_predict(X)
        return self.estimator.score(X, y)


class Scaled(Preprocessed):

    def __init__(self, scaler, estimator):
        super().__init__(estimator)
        self.scaler = scaler

    def _preprocess_X_fit(self, X):
        return self.scaler.fit(X).transform(X)

    def _preprocess_X_predict(self, X):
        return self.scaler.transform(X)


class Polynomial(Preprocessed):
    """
    Unlike sklearn's PolynomialFeatures, this doesn't 
    add intercept column to the data.
    """

    def __init__(self, degree, estimator):
        super().__init__(estimator)
        self.degree = degree

    def _preprocess_X_fit(self, X):
        X = polynomial_features(X, self.degree)
        return X[:,1:] # exclude intercept

    def _preprocess_X_predict(self, X):
        return self._preprocess_X_fit(X) # same as fit


"""
class Scaled(Estimator):

    def __init__(self, scaler, estimator):
        self.estimator = estimator
        self.scaler = scaler

    def fit(self, X, y):
        X = self.scaler.fit(X).transform(X)
        self.estimator = self.estimator.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        X = self.scaler.transform(X)
        return self.estimator.score(X, y)


class Polynomial(Estimator):

    def __init__(self, degree, estimator):
        self.estimator = estimator
        self.degree = degree

    def fit(self, X, y):
        X = polynomial_features(X, self.degree)
        X = X[:,1:] # exclude intercept
        self.estimator = self.estimator.fit(X, y)
        return self

    def predict(self, X):
        X = polynomial_features(X, self.degree)
        X = X[:,1:] # exclude intercept
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        X = polynomial_features(X, self.degree)
        X = X[:,1:] # exclude intercept
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        X = polynomial_features(X, self.degree)
        X = X[:,1:] # exclude intercept
        return self.estimator.score(X, y)
"""