import numpy as np

def entropy(y):
    """ 
    Calculate entropy 
    
    Parameters
    ----------
    y: np.ndarray
        one-hot encoded target data of classification problems
        num of rows (y.shape[0]) is the num of samples, and num of columns (y.shape[1]) is the num of classes
        each value is either 0 or 1, and sum of values in a single row is always 1

    Returns
    -------
    float
        entropy
    """

    if len(y)==0:
        raise ValueError('input must not be empty')

    n_each_class = y.sum(axis=0)
    n_total = y.shape[0]
    p_each_class = n_each_class/n_total
    
    i_each_class = [p*np.log(p) for p in p_each_class if p != 0.0]
    return -np.sum(i_each_class)


def gini_impurity(y):
    """ 
    Calculate gini impurity
    
    Parameters
    ----------
    y: np.ndarray
        one-hot encoded target data of classification problems
        num of rows (y.shape[0]) is the num of samples, and num of columns (y.shape[1]) is the num of classes
        each value is either 0 or 1, and sum of values in a single row is always 1

    Returns
    -------
    float
        gini impurity
    """

    if len(y)==0:
        raise ValueError('input must not be empty')

    n_each_class = y.sum(axis=0)
    n_total = y.shape[0]
    p_each_class = n_each_class/n_total
    
    return 1 - np.sum(p_each_class**2)

def classification_error(y):
    """ 
    Calculate classification error
    
    Parameters
    ----------
    y: np.ndarray
        one-hot encoded target data of classification problems
        num of rows (y.shape[0]) is the num of samples, and num of columns (y.shape[1]) is the num of classes
        each value is either 0 or 1, and sum of values in a single row is always 1

    Returns
    -------
    float
        classification error 
    """

    if len(y)==0:
        raise ValueError('input must not be empty')

    n_max_class = np.max(y.sum(axis=0))
    n_total = y.shape[0]

    return 1 - n_max_class/n_total