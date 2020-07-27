"""
Wrapper classes
Can be used like pipeline.
e.g. clf = Polynomial(2, Scaled(StandardScaler(), LogisticRegression()))

"""

# Author: Shota Horii <sh.sinker@gmail.com>

from mlfs.base_classes import Estimator
from mlfs.utils.transformers import polynomial_features


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
    """
    Unlike sklearn's PolynomialFeatures, this doesn't 
    add intercept column to the data.
    """

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
