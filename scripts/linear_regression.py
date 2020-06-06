"""
Linear Regression

References:
https://qiita.com/fujiisoup/items/e7f703fc57e2dfc441ad

ToDo:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from scripts.preprocessing import add_bias, polynomial_features
from scripts.weight_initialisation import initialise_random, initialise_zero
from scripts.loss_functions import SquareError, L1Regularization, L2Regularization
from scripts.hyperparameter_optimisation import auto_learning_rate_mse

class LinearRegression:

    # solver is analytical or gradient_descent
    
    def __init__(self, solver='analytical', lambda_l2=0, lambda_l1=0, 
                polynomial_degree=1, n_iterations=1000, learning_rate=None):
        self.solver = solver
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.polynomial_degree = polynomial_degree
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.w = None
        self.train_errors = np.array([])
        self.loss = SquareError()
        self.l1 = L1Regularization(lambda_l1)
        self.l2 = L2Regularization(lambda_l2)    

    def _log_mse(self, y, y_pred):
        # though we add l1 & l2 penalties in loss function to minimise, 
        # for monitoring train error, not include penalties.
        mse = np.mean(self.loss(y, y_pred)) # + self.l1(self.w) + self.l2(self.w))
        self.train_errors = np.append(self.train_errors, mse)

    def fit(self, X, y):

        X = polynomial_features(X, self.polynomial_degree)

        self.w = initialise_zero(X.shape[1])

        if self.solver == 'analytical':
            if self.lambda_l1 != 0:
                raise ValueError('L1 reguralisation cannot be solved analytically.')
            
            d = X.shape[1]
            self.w = np.dot( np.dot(np.linalg.pinv( np.dot(X.T, X) + self.lambda_l2 * np.eye(d) ), X.T), y)
            
            # log error
            y_pred = np.dot(self.w, X.T)
            self._log_mse(y, y_pred)

        elif self.solver == 'gradient_descent':
            
            # if learning rate is not given, set automatically
            lr = self.learning_rate if self.learning_rate else auto_learning_rate_mse(X)
            
            for i in range(self.n_iterations):
                y_pred = np.dot(self.w, X.T)
                self._log_mse(y, y_pred)

                grad_w = np.dot(self.loss.gradient(y, y_pred), X) + self.l1.gradient(self.w) + self.l2.gradient(self.w)
                self.w -= lr * grad_w

            # result of the training
            y_pred = np.dot(self.w, X.T)
            self._log_mse(y, y_pred)

        else:
            raise ValueError('solver must be "analytical" or "gradient_descent".')

        return self

    def predict(self, X):

        X = polynomial_features(X, self.polynomial_degree)
        return np.dot(self.w, X.T)

    def train_error(self):
        if len(self.train_errors)==0:
            print('Not trained yet.')
            return None
        else:
            return self.train_errors[-1]


class RidgeRegression(LinearRegression):
    def __init__(self, 
                lambda_l2, 
                solver='analytical', 
                polynomial_degree=1, 
                n_iterations=1000, 
                learning_rate=None):

        super().__init__(solver=solver, 
                lambda_l2=lambda_l2, 
                lambda_l1=0, 
                polynomial_degree=polynomial_degree, 
                n_iterations=n_iterations, 
                learning_rate=learning_rate)

class LassoRegression(LinearRegression):
    def __init__(self, 
                lambda_l1, 
                polynomial_degree=1, 
                n_iterations=1000, 
                learning_rate=None):

        super().__init__(solver='gradient_descent', 
                lambda_l2=0, 
                lambda_l1=lambda_l1, 
                polynomial_degree=polynomial_degree, 
                n_iterations=n_iterations, 
                learning_rate=learning_rate)

class ElasticNetRegression(LinearRegression):
    def __init__(self, 
                alpha,
                l1_ratio,
                polynomial_degree=1, 
                n_iterations=1000, 
                learning_rate=None):

        lambda_l1 = alpha * l1_ratio
        lambda_l2 = alpha * (1 - l1_ratio)
        
        super().__init__(solver='gradient_descent', 
                lambda_l2=lambda_l2, 
                lambda_l1=lambda_l1, 
                polynomial_degree=polynomial_degree, 
                n_iterations=n_iterations, 
                learning_rate=learning_rate)