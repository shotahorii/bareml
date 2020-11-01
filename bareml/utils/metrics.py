"""
Metrics

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from .manipulators import OnehotEncoder


#############################
# Metrics for data impurity #
#############################


def entropy(y, w=None):
    """ 
    Computes entropy 
    
    Parameters
    ----------
    y : np.ndarray (n, c) 
        Target variable of classification problems.
        c = 1 if binary classification. 
        else, c = num of classes for multi classification (one-hot encoded).

    w: np.ndarray (n,)
        weights for y

    Returns
    -------
    float
        entropy
    """
    # if weights are not given, consider weights are all 1.
    w = np.ones(len(y)) if w is None else w

    w_total = np.sum(w)

    if y.ndim==1:
        w_class_1 = np.sum(w[y==1])
        w_each_class = np.array([w_class_1, w_total - w_class_1])
    else:
        w_each_class = np.sum(w[:,None] * y, axis=0)

    p_each_class = w_each_class/w_total    
    i_each_class = [p*np.log(p) for p in p_each_class if p != 0.0]
    return -np.sum(i_each_class)

    # ===== Below is non-weighted implementation =====
    #n_total = len(y)
    #if y.ndim==1:
    #    n_class_1 = y.sum()
    #    n_each_class = np.array([n_class_1, n_total - n_class_1])
    #else:
    #    n_each_class = y.sum(axis=0)
    #p_each_class = n_each_class/n_total    
    #i_each_class = [p*np.log(p) for p in p_each_class if p != 0.0]
    #return -np.sum(i_each_class)


def gini_impurity(y, w=None):
    """ 
    Computes gini impurity
    
    Parameters
    ----------
    y : np.ndarray (n, c) 
        Target variable of classification problems.
        c = 1 if binary classification. 
        else, c = num of classes for multi classification (one-hot encoded).
    
    w: np.ndarray (n,)
        weights for y

    Returns
    -------
    float
        gini impurity
    """
    # if weights are not given, consider weights are all 1.
    w = np.ones(len(y)) if w is None else w

    w_total = np.sum(w)

    if y.ndim==1:
        w_class_1 = np.sum(w[y==1])
        w_each_class = np.array([w_class_1, w_total - w_class_1])
    else:
        w_each_class = np.sum(w[:,None] * y, axis=0)

    p_each_class = w_each_class/w_total
    return 1 - np.sum(p_each_class**2)

    # ===== Below is non-weighted implementation =====
    #n_total = len(y)
    #if y.ndim==1:
    #    n_class_1 = y.sum()
    #    n_each_class = np.array([n_class_1, n_total - n_class_1])
    #else:
    #    n_each_class = y.sum(axis=0)
    #p_each_class = n_each_class/n_total
    #return 1 - np.sum(p_each_class**2)


def classification_error(y, w=None):
    """ 
    Computes classification error
    
    Parameters
    ----------
    y : np.ndarray (n, c) 
        Target variable of classification problems.
        c = 1 if binary classification. 
        else, c = num of classes for multi classification (one-hot encoded).

    w: np.ndarray (n,)
        weights for y

    Returns
    -------
    float
        classification error 
    """
    # if weights are not given, consider weights are all 1.
    w = np.ones(len(y)) if w is None else w

    w_total = np.sum(w)

    if y.ndim==1:
        w_class_1 = np.sum(w[y==1])
        w_max_class = max(w_class_1, w_total - w_class_1)
    else:
        w_max_class = np.max(np.sum(w[:,None] * y, axis=0))
    
    return 1 - w_max_class/w_total

    # ===== Below is non-weighted implementation =====
    #n_total = len(y)
    #if y.ndim==1:
    #    n_class_1 = y.sum()
    #    n_max_class = max(n_class_1, n_total - n_class_1)
    #else:
    #    n_max_class = np.max(y.sum(axis=0))
    #return 1 - n_max_class/n_total


def variance(y, w=None):
    """ 
    Computes variance of the given list of real numbers.
    
    Parameters
    ----------
    y: np.ndarray (n,)

    w: np.ndarray (n, )
        weights for y

    Returns
    -------
    float
        variance
    """
    # if weights are not given, consider weights are all 1.
    w = np.ones(len(y)) if w is None else w

    mu = np.mean(y)
    #var = np.mean(np.power(y-mu,2)) # non-weighted implementation
    var = np.average(np.power(y-mu,2), weights=w)
    return var


def mean_deviation(y, w=None):
    """ 
    Computes mean deviation of the given list of real numbers.
    
    Parameters
    ----------
    y: np.ndarray (n,)

    w: np.ndarray (n, )
        weights for y

    Returns
    -------
    float
        mean deviation
    """

    # if weights are not given, consider weights are all 1.
    w = np.ones(len(y)) if w is None else w

    mu = np.mean(y)
    #md = np.mean(np.abs(y-mu)) # non-weighted implementation
    md = np.average(np.abs(y-mu), weights=w)
    return md


##########################
# Metrics for regression #
##########################


def squared_errors(y, y_pred):
    """ 
    Computes squared error for each sample.
    
    Parameters
    ----------
    y: np.ndarray (1d array)
        Target variable of regression problems.
        Number of elements is the number of data samples. 
    
    y_pred: np.ndarray (1d array)
        Predicted values for the given target variable. 
        Number of elements is the number of data samples. 

    Returns
    -------
    np.ndarray (1d array)
        squared error for each sample
    """
    return np.power(y-y_pred,2)


def absolute_errors(y, y_pred):
    """ 
    Computes absolute error for each sample.
    
    Parameters
    ----------
    y: np.ndarray (1d array)
        Target variable of regression problems.
        Number of elements is the number of data samples. 
    
    y_pred: np.ndarray (1d array)
        Predicted values for the given target variable. 
        Number of elements is the number of data samples. 

    Returns
    -------
    np.ndarray (1d array)
        absolute error for each sample
    """
    return np.abs(y-y_pred)


def absolute_relative_errors(y, y_pred):
    """ 
    Computes absolute relative error for each sample.
    
    Parameters
    ----------
    y: np.ndarray (1d array)
        Target variable of regression problems.
        Number of elements is the number of data samples. 
    
    y_pred: np.ndarray (1d array)
        Predicted values for the given target variable. 
        Number of elements is the number of data samples. 

    Returns
    -------
    np.ndarray (1d array)
        absolute relative error for each sample 
    """
    return np.abs((y-y_pred)/y)


def mse(y, y_pred):
    """ 
    Computes mean squared error.
    
    Parameters
    ----------
    y: np.ndarray (1d array)
        Target variable of regression problems.
        Number of elements is the number of data samples. 
    
    y_pred: np.ndarray (1d array)
        Predicted values for the given target variable. 
        Number of elements is the number of data samples. 

    Returns
    -------
    float
        mean squared error 
    """
    return np.mean(squared_errors(y, y_pred))


def rmse(y, y_pred):
    """ 
    Computes root mean squared error.
    
    Parameters
    ----------
    y: np.ndarray (1d array)
        Target variable of regression problems.
        Number of elements is the number of data samples. 
    
    y_pred: np.ndarray (1d array)
        Predicted values for the given target variable. 
        Number of elements is the number of data samples. 

    Returns
    -------
    float
        root mean squared error 
    """
    return np.sqrt(mse(y, y_pred))


def mae(y, y_pred):
    """ 
    Computes mean absolute error.
    
    Parameters
    ----------
    y: np.ndarray (1d array)
        Target variable of regression problems.
        Number of elements is the number of data samples. 
    
    y_pred: np.ndarray (1d array)
        Predicted values for the given target variable. 
        Number of elements is the number of data samples. 

    Returns
    -------
    float
        mean absolute error 
    """
    return np.mean(absolute_errors(y, y_pred))




def rss(y, y_pred):
    """ residual sum of squares """
    return np.sum(squared_errors(y, y_pred))


def r_squqred(y, y_pred):
    denom = np.sum(np.power(y-y.mean(),2))
    return 1 - rss(y, y_pred)/denom


##############################
# Metrics for classification #
##############################


def _weighted_sum(score, weights, normalise=False):
    """
    Parameters
    ----------
    score: np.ndarray (n,)
    weights: np.ndarray (n,)
    normalise: bool

    Returns
    -------
    float
    """
    if normalise:
        return np.average(score, weights=weights)
    elif weights is not None:
        return score @ weights
    else:
        return score.sum()


def accuracy(y, y_pred, normalise=True, w=None):
    
    if y.ndim == 1: # binary classification
        score = y == y_pred
    else: # multiclass classification
        score = (y == y_pred).all(axis=1)

    return _weighted_sum(score, w, normalise)


def confusion_matrix(y, y_pred):

    onehot = OnehotEncoder()

    if y.ndim == 1: # binary classification
        y = onehot.fit_transform(y)
        y_pred = onehot.transform(y_pred)

    return y.T @ y_pred


def precision_recall_f1(y, y_pred, average='macro'):
    """
    average: str {'macro', 'micro'}
    """
    
    cm = confusion_matrix(y, y_pred)

    if y.ndim == 1: # binary classification
        pr = (cm.diagonal()/cm.sum(axis=0))[1]
        rc = (cm.diagonal()/cm.sum(axis=1))[1]
        f1 = 2*pr*rc / (pr+rc)
    elif average == 'macro': # multiclass - macro average
        pr_per_class = cm.diagonal()/cm.sum(axis=0)
        rc_per_class = cm.diagonal()/cm.sum(axis=1)
        f1_per_class = 2*pr_per_class*rc_per_class / (pr_per_class+rc_per_class)
        pr = np.mean(pr_per_class)
        rc = np.mean(rc_per_class)
        f1 = np.mean(f1_per_class)
    elif average == 'micro': # multiclass - micro average
        pr = rc = f1 = accuracy(y, y_pred)
    else:
        raise ValueError('select "macro" or "micro" average for multiclass.')

    return pr, rc, f1
