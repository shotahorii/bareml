"""
Metrics

References:

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

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
    return np.mean(np.power(y-y_pred,2))

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
    return np.mean(np.abs(y-y_pred))

def rss(y, y_pred):
    """ residual sum of squares """
    return np.sum(np.power(y-y_pred,2))

def r_squqred(y, y_pred):
    denom = np.sum(np.power(y-y.mean(),2))
    return 1 - rss(y, y_pred)/denom

##############################
# Metrics for classification #
##############################

def accuracy(y, y_pred):
    num_errors = np.sum(np.abs(y - y_pred))/2
    return 1 - num_errors/len(y)

def false_positive(y, y_pred):
    pass

def true_positive(y, y_pred):
    pass

def false_negative(y, y_pred):
    pass

def true_negative(y, y_pred):
    pass