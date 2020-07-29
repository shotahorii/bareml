"""
Decision Trees

References:
Shi, H. (2007). Best-first Decision Tree Learning. 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.2862&rep=rep1&type=pdf
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 663-666.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 305-317.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 546-554.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 176-187.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

from abc import ABC, abstractmethod
import math
import random
import numpy as np

from mlfs.base_classes import Regressor, Classifier
from mlfs.utils.metrics import entropy, gini_impurity, variance, mean_deviation, classification_error
from mlfs.utils.transformers import prob2binary

class TreeNode:

    def __init__(self, X, y, w=None, depth=0, split_feature_idx=None, split_threshold=None, 
                impurity_decrease=None, left=None, right=None, value=None):
        
        # training data
        self.X = X
        self.y = y
        self.w = w

        # for all
        self.depth = depth

        # for non-leaf node
        self.split_feature_idx = split_feature_idx
        self.split_threshold = split_threshold
        self.impurity_decrease = impurity_decrease
        self.left = left
        self.right = right
        # for leaf node
        self.value = value

    def to_leaf(self):
        self.left = None
        self.right = None
        self.value = np.mean(self.y, axis=0)        


class Cart:

    def __init__(self, 
                impurity_func=None,
                growth='best', # {'best','depth'}
                max_depth=None, 
                max_leaf_nodes=None, # only works for best-first tree
                min_samples_leaf=1,
                min_impurity_decrease=0,
                max_features=None # for random tree
                ):
        
        self.impurity_func = impurity_func
        self.growth = growth
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.n_samples = None 
        self.tree = None

    def _build_best_first_tree(self, X, y, w):
        self.n_samples = len(y)
        n_leaves = 0

        # initialise root node
        root = self._grow_node(TreeNode(X, y, w))
        nodes = [root]

        while len(nodes) > 0:
            node = nodes.pop(0)

            if node.impurity_decrease <= self.min_impurity_decrease or (self.max_leaf_nodes is not None and n_leaves + len(nodes) + 1 >= self.max_leaf_nodes):
                node.to_leaf()
                for n in nodes: n.to_leaf() # make all nodes into leaf
            elif node.depth == self.max_depth or len(node.left.y) < self.min_samples_leaf or len(node.right.y) < self.min_samples_leaf:
                node.to_leaf() # make this node into leaf
                n_leaves += 1
            else:
                nodes.append(self._grow_node(node.left))
                nodes.append(self._grow_node(node.right))
                # sort nodes by impurity decrease (descending order)
                nodes = sorted(nodes, key=lambda n: n.impurity_decrease, reverse=True)

        return root

    def _build_depth_first_tree(self, X, y, w):

        def build_helper(node):
            if node.impurity_decrease <= self.min_impurity_decrease or node.depth == self.max_depth or len(node.left.y) < self.min_samples_leaf or len(node.right.y) < self.min_samples_leaf:
                node.to_leaf()
            else:
                build_helper(self._grow_node(node.left))
                build_helper(self._grow_node(node.right))

        # init
        self.n_samples = len(y)
        root = self._grow_node(TreeNode(X, y, w))
        build_helper(root)
        
        return root

    def _grow_node(self, node):
        node.split_feature_idx, node.split_threshold, impurity_after_split = self._find_best_split(node.X, node.y, node.w)
        node.impurity_decrease = (1.0*len(node.y)/self.n_samples) * max(self.impurity_func(node.y, node.w) - impurity_after_split, 0)

        left_idx = node.X[:,node.split_feature_idx] < node.split_threshold
        right_idx = np.array([not i for i in left_idx])

        node.left = TreeNode(node.X[left_idx], node.y[left_idx], node.w[left_idx], depth=node.depth+1)
        node.right = TreeNode(node.X[right_idx], node.y[right_idx], node.w[right_idx], depth=node.depth+1)
        return node

    def _compute_impurity(self, y_left, y_right, w_left, w_right):
        """
        Computes total impurity after the given split.
        
        Parameters
        -------
        y_left : np.ndarray (nl, c)
            list of y values which go to the left node by the split
        y_right : np.ndarray (nr, c)
             list of y values which go to the right node by the split

        Returns
        -------
        impurity_after_split : float
            value of the impurity after the split
        """
        left_ratio = 1.0*np.sum(w_left)/(np.sum(w_left) + np.sum(w_right))
        right_ratio = 1.0*np.sum(w_right)/(np.sum(w_left) + np.sum(w_right))

        if left_ratio == 0 or right_ratio == 0:
            impurity_after_split = np.inf
        else:
            left_impurity = self.impurity_func(y_left, w_left)
            right_impurity = self.impurity_func(y_right, w_right)
            impurity_after_split = left_impurity * left_ratio + right_impurity * right_ratio
        
        return impurity_after_split
        
    def _find_best_split_threshold(self, x_feature, y, w):
        """
        Finds the best split threshold value for the given feature
        
        Parameters
        ----------
        x_feature : np.ndarray (n, )
        y : np.ndarray (n, c) 

        Returns
        -------
        best_threshold: float
        impurity_after_split: float
        """
        
        min_impurity = np.inf
        best_threshold = np.inf
        
        possible_thresholds = set(x_feature)
        
        for threshold in possible_thresholds:
            left_idx = x_feature < threshold
            right_idx = np.array([not i for i in left_idx])
            impurity_after_split = self._compute_impurity(y[left_idx], y[right_idx], w[left_idx], w[right_idx])
            
            if impurity_after_split < min_impurity:
                min_impurity = impurity_after_split
                best_threshold = threshold
        
        return best_threshold, impurity_after_split

    def _find_best_split(self, X, y, w):
        """
        Finds the best split across all the input features
        
        Parameters
        -------
        X : np.ndarray (n, d)       
        y : np.ndarray (n, c) 

        Returns
        -------
        best_split_feature_idx: int
        best_split_threshold: float
        max_decrease: float
        """
        
        # init 
        min_impurity = np.inf
        best_split_feature_idx = 0
        best_split_threshold = np.inf # all goes to the left
        
        for feature_idx in self._search_scope(X):
            x_feature = X[:,feature_idx]
            threshold, impurity_after_split = self._find_best_split_threshold(x_feature, y, w)
            if impurity_after_split < min_impurity:
                best_split_feature_idx = feature_idx
                best_split_threshold = threshold
                min_impurity = impurity_after_split
        
        return best_split_feature_idx, best_split_threshold, min_impurity

    def _search_scope(self, X):
        """ Returns indices of features to serch best split."""
        n_features = X.shape[1]

        # normal tree
        if self.max_features is None:
            return range(n_features)

        # random tree
        if self.max_features == 'sqrt':
            n_pick = round(math.sqrt(n_features))
        elif self.max_features == 'log2':
            n_pick = max(1, round(math.log2(n_features)))
        elif isinstance(self.max_features, int):
            n_pick = min(n_features, max(1, self.max_features))
        else:
            raise ValueError('Invalid input for max_features.')
        
        indices = random.sample(range(n_features), n_pick)
        return indices

    def fit(self, X, y, w=None):
        """
        Build the decision tree fitting the training data.
        
        Parameters
        -------
        X : np.ndarray (n, d)
        y : np.ndarray (n, c) 
        """

        # if weights are not given, consider weights are all 1.
        w = np.ones(len(y)) if w is None else w

        if self.growth == 'best':
            self.tree = self._build_best_first_tree(X, y, w)
        elif self.growth == 'depth':
            self.tree = self._build_depth_first_tree(X, y, w)
        else:
            raise ValueError('Growth needs to be "best" or "depth".')
            
        return self

    def _retrieve_tree(self, X, node):

        num_x = len(X) # number of samples to predict

        if node.value is not None: # leaf
            if node.value.ndim == 0: # binary classification or regression
                pred = np.zeros(num_x) + node.value
            
            else: # multi classification
                num_y = node.value.shape[0] # number of classes
                pred = np.zeros((num_x, num_y)) + node.value

        else: # non-leaf node
            left_idx = X[:,node.split_feature_idx] < node.split_threshold
            right_idx = np.array([not i for i in left_idx])

            left_pred = self._retrieve_tree(X[left_idx], node.left)
            right_pred = self._retrieve_tree(X[right_idx], node.right)

            if left_pred.ndim == 1:
                pred = np.zeros(num_x)
            else:
                num_y = left_pred.shape[1]
                pred = np.zeros((num_x, num_y))

            pred[left_idx] = left_pred
            pred[right_idx] = right_pred
            
        return pred

    def predict(self, X):
        return self._retrieve_tree(X, self.tree)


class DecisionTreeClassifier(Cart, Classifier):

    def __init__(self, 
                criterion='gini',
                growth='best', # {'best','depth'}
                max_depth=None, 
                max_leaf_nodes=None, # only works for best-first tree
                min_samples_leaf=1,
                min_impurity_decrease=0):
        
        if criterion == 'gini':
            impurity_func = gini_impurity
        elif criterion == 'entropy':
            impurity_func = entropy
        else:
            raise ValueError('Criterion needs to be either "gini" or "entropy".')

        super().__init__(impurity_func=impurity_func, 
                         growth=growth, 
                         max_depth=max_depth, 
                         max_leaf_nodes=max_leaf_nodes,
                         min_samples_leaf = min_samples_leaf,
                         min_impurity_decrease=min_impurity_decrease)

    def fit(self, X, y, w=None):
        X, y, w = self._validate_Xyw(X, y, w)
        return super().fit(X, y, w)

    def predict_proba(self, X):
        X = self._validate_X(X)
        return super().predict(X)

    def predict(self, X):
        return prob2binary(self.predict_proba(X))


class DecisionTreeRegressor(Cart, Regressor):

    def __init__(self, 
                criterion='mse',
                growth='best', # {'best','depth'}
                max_depth=None, 
                max_leaf_nodes=None, # only works for best-first tree
                min_samples_leaf=1,
                min_impurity_decrease=0):
        
        if criterion == 'mse':
            impurity_func = variance
        elif criterion == 'mae':
            impurity_func = mean_deviation
        else:
            raise ValueError('Criterion needs to be either "mse" or "mae".')

        super().__init__(impurity_func=impurity_func, 
                         growth=growth, 
                         max_depth=max_depth, 
                         max_leaf_nodes=max_leaf_nodes,
                         min_samples_leaf = min_samples_leaf,
                         min_impurity_decrease=min_impurity_decrease)

    def fit(self, X, y, w=None):
        X, y, w = self._validate_Xyw(X, y, w)
        return super().fit(X, y, w)

    def predict(self, X):
        X = self._validate_X(X)
        return super().predict(X)


