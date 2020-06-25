"""
Decision Trees

References:
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 663-666.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 305-317.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 546-554.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 176-187.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
from mlfs.utils.metrics import entropy, gini_impurity, variance, mean_deviation

class DecisionTree:
    """
    Super class of DecisionTreeClassifier and DecisionTreeRegressor.

    Parameters
    ----------
    is_regression: bool
        if this tree is for regression. if false, for classification
    
    impurity_func: function
        function used to measure the impurity
        choices are: entropy, gini_impurity, variance, mean_deviation

    max_depth: int
        maximum depth that the tree can grow
    
    min_impurity_decrease: float
        if the impurity decrease is smaller than this, 
        tree doesn't split.

    N: int
        the total number of data used for training the tree

    depth: int
        current depth of the tree
    """
    
    def __init__(
        self, 
        is_regression=None, 
        impurity_func=None,
        max_depth=None, 
        min_impurity_decrease=None, 
        N=None,
        depth=1):
        
        # parameters to define the behaviour of the entire tree (common across all nodes in the tree)
        self.is_regression = is_regression
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.N = N
        self.impurity_func = impurity_func
        
        # parameters for each node
        self.depth = depth
        self.impurity = None
        self.left = None
        self.right = None
        
        # parameters about node split
        self.impurity_decrease = None
        self.split_feature_idx = None
        self.split_threshold = None
        
        # a parameter for leaf nodes
        self.predictor = None
        
        
    def _create_node(self):
        """
        Create a node with +1 depth.
        """
        return DecisionTree(is_regression=self.is_regression,
                            impurity_func=self.impurity_func, 
                            max_depth=self.max_depth, 
                            min_impurity_decrease=self.min_impurity_decrease,
                            N=self.N,
                            depth=self.depth+1)

    def _compute_impurity_decrease(self, y_left, y_right):
        """
        Computes impurity decrease (aka information gain when impurity 
        measure is entropy) for the given split.
        
        Parameters
        -------
        y_left : np.ndarray
            list of y values which go to the left node by the split
            
        y_right : np.ndarray
             list of y values which go to the right node by the split

        Returns
        -------
        decrease : float
            value of the impurity decrease
        """

        w_left = 1.0*len(y_left)/(len(y_left) + len(y_right))
        w_right = 1.0*len(y_right)/(len(y_left) + len(y_right))

        if w_left == 0 or w_right == 0:
            decrease = 0
        else:
            left_impurity = self.impurity_func(y_left)
            right_impurity = self.impurity_func(y_right)
            decrease = self.impurity - (left_impurity*w_left + right_impurity*w_right)
        
        return decrease
        
    
    def _find_best_split_threshold(self, x_feature, y):
        """
        Finds the best split threshold value of the given feature
        
        Parameters
        -------
        x_feature : np.ndarray (1d array)
            a predictor variable (feature) to be checked
            number of elements is the number of data samples
            
        y : np.ndarray
            Target variable of classification or Regression problems.
            This can be a 1d array indicating 0/1 for a binary classification
            or real number for a regression.
            Or a multi-dimensional array, indicating a one-hot encoded 
            target variable for a multi-class classification. 

            When 1d array:
                Num of elements is the num of samples. 
                Each value is 0 or 1, if classification
                Each value is a real number if regression

            When multi-dimensional array
                Num of rows (y.shape[0]) is the num of samples.
                Num of columns (y.shape[1]) is the num of classes.
                Each value is either 0 or 1.
                Sum of values in a single row is always 1.

        Returns
        -------
        max_decrease: float
            maximum decrease of impurity obtained by the split using 
            this feature and the best threshold

        best_threshold: float
            the best threshold to split the data 
        """
        
        max_decrease = 0
        best_threshold = np.inf
        
        possible_thresholds = set(x_feature)
        
        for threshold in possible_thresholds:
            left_idx = x_feature < threshold
            right_idx = np.array([not i for i in left_idx])
            decrease = self._compute_impurity_decrease(y[left_idx], y[right_idx])
            
            if decrease > max_decrease:
                max_decrease = decrease
                best_threshold = threshold
        
        return max_decrease, best_threshold

        
    def _find_best_split(self, X, y):
        """
        Finds the best split across all the input features
        
        Parameters
        -------
        X : np.ndarray (can be 1d array if there's only 1 feature)
            predictor variables (features) to be checked
            num of row (X.shape[0]) is the num of data samples
            num of columns (X.shape[1]) is the num of features
            
        y : np.ndarray
            Target variable of classification or Regression problems.
            This can be a 1d array indicating 0/1 for a binary classification
            or real number for a regression.
            Or a multi-dimensional array, indicating a one-hot encoded 
            target variable for a multi-class classification. 

            When 1d array:
                Num of elements is the num of samples. 
                Each value is 0 or 1, if classification
                Each value is a real number if regression

            When multi-dimensional array
                Num of rows (y.shape[0]) is the num of samples.
                Num of columns (y.shape[1]) is the num of classes.
                Each value is either 0 or 1.
                Sum of values in a single row is always 1.

        Returns
        -------
        best_split_feature_idx:
            index of the best feature to split the data
        
        best_split_threshold: float
            the best threshold to split the data

        max_decrease: float
            maximum decrease of impurity obtained by the split using 
            this feature and the best threshold
        """
        
        # init 
        max_decrease = 0
        best_split_feature_idx = 0
        best_split_threshold = np.inf # all goes to the left
        
        if X.ndim == 1: # there's only one feature in the data
            decrease, threshold = self._find_best_split_threshold(X, y)
            if decrease > max_decrease:
                best_split_threshold = threshold

        else: # more than 2 features in the data
            n_features = X.shape[1]
            for feature_idx in range(n_features):
                x_feature = X[:,feature_idx]
                decrease, threshold = self._find_best_split_threshold(x_feature, y)
                if decrease > max_decrease:
                    best_split_feature_idx = feature_idx
                    best_split_threshold = threshold
        
        return best_split_feature_idx, best_split_threshold, max_decrease
            

    def fit(self, X, y):
        """
        Build the decision tree fitting the training data.
        
        Parameters
        -------
        X : np.ndarray (can be 1d array if there's only 1 feature)
            predictor variables (features) to be checked
            num of row (X.shape[0]) is the num of data samples
            num of columns (X.shape[1]) is the num of features
            
        y : np.ndarray
            Target variable of classification or Regression problems.
            This can be a 1d array indicating 0/1 for a binary classification
            or real number for a regression.
            Or a multi-dimensional array, indicating a one-hot encoded 
            target variable for a multi-class classification. 

            When 1d array:
                Num of elements is the num of samples. 
                Each value is 0 or 1, if classification
                Each value is a real number if regression

            When multi-dimensional array
                Num of rows (y.shape[0]) is the num of samples.
                Num of columns (y.shape[1]) is the num of classes.
                Each value is either 0 or 1.
                Sum of values in a single row is always 1.

        Returns
        -------
        self: DecisionTree
        """

        if self.is_regression and y.ndim > 1:
            raise ValueError("Target variable for regression must be in R^1 space.")
        
        # if this is a root node, store the size of entire training data set
        if self.depth == 1:
            self.N = len(y)
        
        # calculate this node's impurity 
        self.impurity = self.impurity_func(y)
        
        # find the best split
        best_split_feature_idx, best_split_threshold, max_decrease = self._find_best_split(X, y)
        
        # split the data by using the best split information
        if X.ndim == 1:
            left_idx = X < best_split_threshold
        else:
            left_idx = X[:,best_split_feature_idx] < best_split_threshold
        right_idx = np.array([not i for i in left_idx])
        
        # store the split information
        self.split_feature_idx = best_split_feature_idx
        self.split_threshold = best_split_threshold
        self.impurity_decrease = max_decrease * (1.0*len(y)/self.N)
        
        # update 
        if self.max_depth is not None and self.depth >= self.max_depth:
            self.predictor = np.mean(y, axis=0)
        elif self.min_impurity_decrease is not None and self.impurity_decrease < self.min_impurity_decrease:
            self.predictor = np.mean(y, axis=0)
        elif left_idx.sum()==0 or right_idx.sum()==0:
            self.predictor = np.mean(y, axis=0)
        else:
            # fit left
            self.left = self._create_node()
            self.left.fit(X[left_idx], y[left_idx])
            # fit right
            self.right = self._create_node()
            self.right.fit(X[right_idx], y[right_idx])
            
        return self
 

    def predict(self, X):
        """
        Predict based on the trained tree. 
        
        Parameters
        -------
        X : np.ndarray (can be 1d array if there's only 1 feature)
            predictor variables (features) to be used for the prediction.
            num of row (X.shape[0]) is the num of data samples
            num of columns (X.shape[1]) is the num of features
            
        Returns
        -------
        pred: np.ndarray
            Predicted target variable of classification or Regression problems.
            This can be a 1d array indicating 0/1 for a binary classification
            or real number for a regression.
            Or a multi-dimensional array, indicating a one-hot encoded 
            target variable for a multi-class classification. 

            When 1d array:
                Num of elements is the num of samples. 
                Each value is 0 or 1, if classification
                Each value is a real number if regression

            When multi-dimensional array
                Num of rows (y.shape[0]) is the num of samples.
                Num of columns (y.shape[1]) is the num of classes.
                Each value is either 0 or 1.
                Sum of values in a single row is always 1.

        """
        
        if self.left is None or self.right is None: # this is a leaf node
            num_x = len(X) # number of samples to predict
            if self.predictor.ndim == 0:
                pred = np.zeros(num_x) + self.predictor
                if not self.is_regression: # classification: convert to 0/1
                    pred = np.round(pred).astype(int)
            else:
                num_y = self.predictor.shape[0] # number of y to predict (= number of classes if it's classification)
                pred = np.zeros((num_x, num_y)) + self.predictor
                if not self.is_regression: # classification
                    cls_pred = np.zeros((num_x, num_y),dtype=int)
                    for i_sample, i_class in enumerate(np.argmax(pred,axis=1)):
                        cls_pred[i_sample, i_class] = 1
                    pred = cls_pred
        
        else: # this isn't a leaf node   

            if X.ndim == 1:
                left_idx = X < self.split_threshold
            else:
                left_idx = X[:,self.split_feature_idx] < self.split_threshold

            right_idx = np.array([not i for i in left_idx])

            left_pred = self.left.predict(X[left_idx])
            right_pred = self.right.predict(X[right_idx])

            num_x = len(X) # number of samples to predict

            if left_pred.ndim == 1:
                pred = np.zeros(num_x)
            else:
                num_y = left_pred.shape[1]
                pred = np.zeros((num_x, num_y))

            pred[left_idx] = left_pred
            pred[right_idx] = right_pred
            
        return pred

class DecisionTreeClassifier(DecisionTree):
    """
    Decision tree for classification. 

    Parameters
    ----------
    criterion: string
        criterion to be used for measuring the impurity.
        either "gini" or "entropy"

    max_depth: int
        maximum depth that the tree can grow
    
    min_impurity_decrease: float
        if the impurity decrease is smaller than this, 
        tree doesn't split.
    """

    def __init__(
        self, 
        criterion='gini',
        max_depth=None, 
        min_impurity_decrease=None):

        if criterion == 'gini':
            impurity_func = gini_impurity
        elif criterion == 'entropy':
            impurity_func = entropy
        else:
            raise ValueError('metric parameter needs to be either "gini" or "entropy".')
        
        super().__init__(
            is_regression=False,
            impurity_func=impurity_func,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            N=None,
            depth=1
        )

class DecisionTreeRegressor(DecisionTree):
    """
    Decision tree for regression. 

    Parameters
    ----------
    criterion: string
        criterion to be used for measuring the impurity.
        either "mse" or "mae"

    max_depth: int
        maximum depth that the tree can grow
    
    min_impurity_decrease: float
        if the impurity decrease is smaller than this, 
        tree doesn't split.
    """

    def __init__(
        self, 
        criterion='mse',
        max_depth=None, 
        min_impurity_decrease=None):

        if criterion == 'mse':
            impurity_func = variance
        elif criterion == 'mae':
            impurity_func = mean_deviation
        else:
            raise ValueError('metric parameter needs to be either "mse" or "mae".')
        
        super().__init__(
            is_regression=True,
            impurity_func=impurity_func,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            N=None,
            depth=1
        )