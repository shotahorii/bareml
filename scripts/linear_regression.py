"""
Linear Regression

References:
https://qiita.com/fujiisoup/items/f2fe3b508763b0cc6832

ToDo:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from scripts.preprocessing import polynomial_features
from scripts.solvers import PInv, LassoISTA, LeastSquareGD
from scripts.metrics import mean_square_error

class LinearRegression:

    # solver is pinv or gradient_descent
    
    def __init__(self, solver='pinv', alpha_l1=0, alpha_l2=0, polynomial_degree=1, 
        max_iterations=1000, tol=1e-4, learning_rate=None):
        
        # some obvious warnings.
        if solver == 'pinv' and alpha_l1 != 0:
            raise ValueError('L1 reguralisation cannot be solved analytically.')

        if solver == 'gradient_descent' and alpha_l1 != 0:
            raise ValueError('"gradient_descent" solver here is the simple one, which \
                cannot solve Lasso. Use "lasso" solver instead, which is ISTA.')
        
        self.solver = solver
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2

        self.polynomial_degree = polynomial_degree

        self.max_iterations = max_iterations
        self.tol = tol
        self.learning_rate = learning_rate

        self.w = None
        self.train_error = None

    def fit(self, X, y):

        X = polynomial_features(X, self.polynomial_degree)

        if self.solver == 'pinv':
            pinv = PInv(alpha=self.alpha_l2)
            self.w = pinv.solve(X, y)

        elif self.solver == 'lasso':
            lasso = LassoISTA(self.alpha_l1, self.max_iterations, self.tol)
            self.w = lasso.solve(X, y)

        elif self.solver == 'gradient_descent':
            gd = LeastSquareGD(self.alpha_l2, self.max_iterations, self.tol, self.learning_rate)
            self.w = gd.solve(X,y)

        else:
            raise ValueError('"' + self.solver + '" solver not found. ' + \
                'solver must be "pinv", "lasso" or "gradient_descent".')

        # log training error
        y_pred = np.dot(self.w, X.T)
        self.train_error = mean_square_error(y, y_pred)

        return self

    def predict(self, X):

        X = polynomial_features(X, self.polynomial_degree)
        return np.dot(self.w, X.T)


class RidgeRegression(LinearRegression):
    def __init__(self, 
                alpha, 
                solver='pinv', 
                polynomial_degree=1, 
                max_iterations=1000, 
                tol=1e-4,
                learning_rate=None):

        super().__init__(solver=solver, 
                alpha_l2=alpha, 
                polynomial_degree=polynomial_degree, 
                max_iterations=max_iterations, 
                tol=tol,
                learning_rate=learning_rate)


class LassoRegression(LinearRegression):
    def __init__(self, 
                alpha, 
                polynomial_degree=1, 
                max_iterations=1000, 
                tol=1e-4):

        super().__init__(solver='lasso', 
                alpha_l1=alpha, 
                polynomial_degree=polynomial_degree, 
                max_iterations=max_iterations, 
                tol=tol)


class ElasticNetRegression(LinearRegression):
    def __init__(self, 
                alpha,
                l1_ratio,
                polynomial_degree=1, 
                max_iterations=1000, 
                learning_rate=None):

        alpha_l1 = alpha * l1_ratio
        alpha_l2 = alpha * (1 - l1_ratio)
        
        super().__init__(solver='NOT IMPLEMENTED YET!!!', 
                alpha_l1=alpha_l1, 
                alpha_l2=alpha_l2, 
                polynomial_degree=polynomial_degree, 
                max_iterations=max_iterations, 
                learning_rate=learning_rate)