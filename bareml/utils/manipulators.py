"""
Data manipulator classes and functions

Author: Shota Horii <sh.sinker@gmail.com>

References:
"""

import math
from abc import ABCMeta, abstractmethod
from itertools import combinations_with_replacement

import numpy as np

from .misc import flatten


########## Scalers and Encoders ##########


class Transform(metaclass=ABCMeta):
    """ 
    A base class for transform classes
    including scalers and encoders
    """

    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass


class StandardScaler(Transform):
    """ 
    Feature Scaler (Standardisation)
    X -> (X - mu) / std 
    """

    def __init__(self):
        self.mu = None
        self.s = None
    
    def fit(self, X):
        """
        Compute and store mean and standard deviation 
        for each column (feature) of the given matrix X.

        Parameters
        ----------
        X: np.array (n, d) float in (-inf, inf)
            n: number of samples 
            d: number of features
        """
        self.mu = np.mean(X, axis=0)
        self.s = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Transform the given matrix X using the stored mean and std. 
        X -> (X - mu) / std

        Parameters
        ----------
        X: np.array (n, d) float in (-inf, inf)
            n: number of samples 
            d: number of features
        """
        y = (X - self.mu) / self.s
        return y

    def fit_transform(self, X):
        """
        Perform fit and transform.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Transform given X into the original representation.
        """
        y = self.s * X + self.mu
        return y


class OnehotEncoder(Transform):
    """
    Onehot encoder.
    For example:
    labels_ = ['male', 'female', 'unknown']
    transform(['male','unknown','female]) -> [[1,0,0], [0,0,1], [0,1,0]]
    inverse_transform([[0,1,0],[0,0,1]]) -> ['female', 'unknown']
    """

    def __init__(self):
        self.labels_ = None

    def _validate_X(self, X):
        """ validate input X to be encoded """
        X = np.array(X)

        if X.ndim != 1:
            raise ValueError('input array needs to be 1d')

        return X

    def _validate_y(self, y):
        y = np.array(y)

        if y.ndim != 2:
            raise ValueError('input array needs to be 2d')
        elif not np.issubdtype(y.dtype, np.number):
            raise ValueError('Data type of y needs to be numeric.')
        elif not np.isin(y, [0,1]).all():
            raise ValueError('Element in y needs to be 0 or 1.')
        elif (y.sum(axis=1)>1).any():
            raise ValueError('Each sample cannot be asigned to more than 1 class.')
        
        return y

    def fit(self, X):
        """ 
        fit
        
        Parameters
        ----------
        X: np.array (n,)
            n: number of samples 

        Returns
        -------
        self
        """
        X = self._validate_X(X)
        X_uniq = np.unique(X)
        
        if len(X_uniq) == 1:
            raise ValueError('Cannot fit with only 1 unique value array.')
        else:
            self.labels_ = np.unique(X)

        return self

    def transform(self, X):
        """ 
        Encode nominal values to one-hot encoding 
        
        Parameters
        ----------
        X: np.array (n,)
            n: number of samples 

        Returns
        -------
        y: np.array (n, c) int/float in {0, 1}
            n: number of samples 
            c: number of classes
        """
        X = self._validate_X(X)

        if self.labels_ is None:
            raise ValueError('labels needed to be fitted.')
        elif not set(X).issubset(self.labels_):
            raise ValueError('input array includes value(s) not in the labels.')
        
        y = np.array([(y==v).astype(int) for v in self.labels_]).T 
        return y

    def fit_transform(self, X):
        """ fit and transform """
        return self.fit(X).transform(X)

    def inverse_transform(self, y):
        """ 
        Decodes one-hot encoding to nominal values 

        Parameters
        ----------
        y: np.array (n, c) int/float in {0, 1}
            n: number of samples 
            c: number of classes

        Returns
        -------
        X: np.array (n,)
            n: number of samples 
        """
        y = self._validate_y(y)

        if self.labels_ is None:
            self.labels_ = np.arange(y.shape[1])

        return np.array([self.labels_[i] for i in np.argmax(y, axis=1)])


class BinaryEncoder(Transform):
    """
    Binary encoder
    """

    def __init__(self, neg_val=-1, pos_val=1):
        self.labels_ = None

        if neg_val == pos_val:
            raise ValueError('neg_val and pos_val cannot be the same value.')

        self.neg_val = neg_val
        self.pos_val = pos_val

    def _validate_X(self, X):
        """ validate input X to be encoded """
        X = np.array(X)

        if X.ndim != 1:
            raise ValueError('input array needs to be 1d')

        return X

    def _validate_y(self, y):
        y = np.array(y)

        if y.ndim != 1:
            raise ValueError('input array needs to be 1d')
        elif not np.isin(y, [self.neg_val,self.pos_val]).all():
            raise ValueError('Element in y needs to be one of the 2 designated values.')
        
        return y

    def fit(self, X):
        """ 
        fit
        
        Parameters
        ----------
        X: np.array (n,)
            n: number of samples 

        Returns
        -------
        self
        """
        X = self._validate_X(X)

        if len(X_uniq) != 2:
            raise ValueError('input needs to have exact 2 classes.')
        else:
            self.labels_ = np.unique(X)

        return self

    def transform(self, X):
        """ 
        Encode real values to binary encoding 
        
        Parameters
        ----------
        X: np.array (n,)
            n: number of samples 

        Returns
        -------
        y: np.array (n,) int in {self.neg_val, self.pos_val}
            n: number of samples
        """
        X = self._validate_X(X)

        if self.labels_ is None:
            raise ValueError('labels needed to be fitted.')
        elif not set(X).issubset(self.labels_):
            raise ValueError('input array includes value(s) not in the labels.')

        y = np.array([self.neg_val if x==self.labels_[0] else self.pos_val for x in X])
        return y

    def fit_transform(self, X):
        """ fit and transform """
        return self.fit(X).transform(X)

    def inverse_transform(self, y):
        """ 
        Decode

        Parameters
        ----------
        y: np.array (n,) int/float in {self.neg_val, self.pos_val}
            n: number of samples 

        Returns
        -------
        X: np.array (n,)
            n: number of samples 
        """
        y = self._validate_y(y)

        if self.labels_ is None:
            self.labels_ = (self.neg_val, self.pos_val) # no encoding/decoding

        X = np.array([self.labels_[0] if yi==self.neg_val else self.labels_[1] for yi in y])
        return X    


########## Other functions ##########


def real2binary(y, threshold=0.5, inclusive=True):
    """
    Convert real values (-inf,inf) -> binary values {0,1}

    Parameters
    ----------
    y: np.array (n,c) float/int (-inf,inf)

    threshold: float (-inf, inf)
        value greater than this is converted into 1, otherwise 0

    inclusive: bool
        if True, equal to threshold -> 1, else -> 0

    Returns 
    -------
    np.array (n,c) int {0,1}
    """
    if inclusive:
        return (y >= threshold).astype(int)
    else:
        return (y > threshold).astype(int)


def binary2sign(y):
    """
    Convert binary values {0,1} -> signs {-1,1}

    Parameters
    ----------
    y: np.array (n,c) float/int {0,1}

    Returns
    -------
    np.array (n,c) int {-1,1}
    """
    return np.sign(y-0.5).astype(int)


def sign2binary(y, zero_as_plus=False):
    """
    Convert signs {-x,x} -> binary values {0,1}

    Parameters
    ----------
    y: np.array (n,c) float/int (-inf,inf)

    zero_as_plus: bool
        if True, convert 0 -> 1, else 0 -> 0

    Returns
    -------
    np.array (n,c) int {0,1}
    """
    if zero_as_plus:
        return (y >= 0).astype(int)
    else:
        return (y > 0).astype(int)

def real2sign(y, zero_as_plus=True):
    """
    Convert real values (-inf,inf) -> signs {-1,1}

    Parameters
    ----------
    y: np.array (n,c) float/int (-inf,inf)

    zero_as_plus: bool
        if True, convert 0 -> 1, else 0 -> -1

    Returns
    -------
    np.array (n,c) int {-1,1}
    """
    if y.ndim == 0: # case when y is a np scalar
        if zero_as_plus:
            return 1 if y==0 else np.sign(y).astype(int) 
        else:
            return -1 if y==0 else np.sign(y).astype(int)

    if zero_as_plus:
        return np.where(y==0, 1, np.sign(y).astype(int))
    else:
        return np.where(y==0, -1, np.sign(y).astype(int)) 

def prob2binary(y):
    """
    Convert probability to binary data. 
    For example, [0.6, 0.2, 0.8] -> [1, 0, 1]
    Also, [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]] -> [[0, 1, 0], [0, 0, 1]]

    Parameters
    ----------
    y: np.ndarray (n,c) float [0,1]

    Returns
    -------
    np.ndarray (n,c) int {0,1}
    """
    if y.ndim <= 1:
        return np.round(y).astype(int)
    else:
        # avoid [[0.333, 0.333, 0.333], [0.2, 0.4, 0.4]] -> [[1, 1, 1], [0, 1, 1]]
        # instead, [[0.333, 0.333, 0.333], [0.2, 0.4, 0.4]] -> [[1, 0, 0], [0, 1, 0]]
        y_bin = np.zeros_like(y)
        y_bin[np.arange(len(y)), y.argmax(axis=1)] = 1 
        return y_bin


def add_intercept(X):
    """
    Add intercept (a column filled with 1. ) to the feature matrix.
    Parameters
    ----------
    X: np.array (n, d) float in (-inf, inf)
        n: number of samples 
        d: number of features
    """
    return np.insert(X, 0, 1, axis=1)


def polynomial_features(X, degree):
    """
    Generates a new feature matrix consisting of all polynomial combinations 
    of the features with degree less than or equal to the given degree. 
    e.g. X=[a, b], degree=2 then [1, a, b, a^2, ab, b^2]

    Parameters
    ----------
    X: np.array (n, d) float in (-inf, inf)
        n: number of samples 
        d: number of features

    degree: int in {1,2,3,...} 
    """
    n_features = X.shape[1]
    index_combinations = flatten([combinations_with_replacement(range(n_features), i) for i in range(0,degree+1)])
    
    return np.array([np.prod(X[:, comb], axis=1) for comb in index_combinations]).T  

