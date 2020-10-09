"""
Transformer classes and functions

References:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement

import numpy as np

from bareml.utils.misc import flatten


########## Scalers and Encoders ##########


class Scaler(ABC):
    """ A base class for scalers """

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass


class Encoder(ABC):
    """ A base class for encoders """

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class StandardScaler(Scaler):
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
        return (X - self.mu) / self.s


class OnehotEncoder(Encoder):
    """
    Onehot encoder.
    For example:
    code = ['male', 'female', 'unknown']
    encode(['male','unknown','female]) -> [[1,0,0], [0,0,1], [0,1,0]]
    decode([[0,1,0],[0,0,1]]) -> ['female', 'unknown']

    Parameters
    ----------
    code: np.array (c,)
        an array of unique labels to be encoded into onehot expression
        c: number of labels
    """

    def __init__(self, code=None):
        self.code = self._codable(code)

    def _codable(self, code):
        """ validate if the given code is valid. """

        if code is None:
            return code

        if len(code) != len(set(code)):
            raise ValueError('code needs to be an array of unique values.')
        elif len(code) < 2:
            raise ValueError('code needs to contain more than 1 values.')

        return code

    def _encodable(self, y):
        """ validate if the given y is encodable. """

        if not set(y).issubset(self.code):
            raise ValueError('input array includes value(s) not in the code.')

        return y

    def encode(self, y):
        """ 
        Encode nominal values to one-hot encoding 
        
        Parameters
        ----------
        y: np.array (n,)
            n: number of samples 

        Returns
        -------
        Y: np.array (n, c) int/float in {0, 1}
            n: number of samples 
            c: number of classes
        """

        if self.code is None:
            self.code = self._codable(np.unique(y))
        else:
            y = self._encodable(y)

        return np.array([(y==v).astype(int) for v in self.code]).T 

    def decode(self, Y):
        """ 
        Decodes one-hot encoding to nominal values 

        Parameters
        ----------
        Y: np.array (n, c) int/float in {0, 1}
            n: number of samples 
            c: number of classes

        Returns
        -------
        y: np.array (n,)
            n: number of samples 
        """

        if self.code is None:
            self.code = np.arange(Y.shape[1])

        return np.array([self.code[i] for i in np.argmax(Y, axis=1)])


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

