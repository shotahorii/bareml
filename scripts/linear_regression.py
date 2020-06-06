"""
Linear Regression

References:

ToDo:
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from scripts.preprocessing import add_bias, polynomial_features
from scripts.weight_initialisation import initialise_random, initialise_zero
from scripts.loss_functions import SquareError, L1Regularization, L2Regularization

class LinearRegression:

    # solver is analytical or gradient_descent
    
    def __init__(self, solver='analytical', lambda_l2=0, lambda_l1=0, 
                polynomial_degree=1, n_iterations=1000, learning_rate=0.0001):
        self.solver = solver
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.polynomial_degree = polynomial_degree
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.w = None

    def fit(self, X, y):

        X = polynomial_features(X, self.polynomial_degree)

        self.w = initialise_zero(X.shape[1])

        if self.solver == 'analytical':
            if self.lambda_l1 != 0:
                raise ValueError('L1 reguralisation cannot be solved analytically.')
            
            d = X.shape[1]
            self.w = np.dot( np.dot(np.linalg.pinv( np.dot(X.T, X) + self.lambda_l2 * np.eye(d) ), X.T), y)

        elif self.solver == 'gradient_descent':
            loss = SquareError()
            l1 = L1Regularization(self.lambda_l1)
            l2 = L1Regularization(self.lambda_l2)

            for i in range(self.n_iterations):
                y_pred = np.dot(self.w, X.T)
                
                # no need of calculating error everytime...
                #mse = np.mean(loss(y, y_pred) + l1(self.w) + l2(self.w))
                grad_w = np.dot(loss.gradient(y, y_pred), X) + l1.gradient(self.w) + l2.gradient(self.w)
                self.w -= self.learning_rate * grad_w

        else:
            raise ValueError('solver must be "analytical" or "gradient_descent".')

        return self

    def predict(self, X):

        X = polynomial_features(X, self.polynomial_degree)
        return np.dot(self.w, X.T)


class RidgeRegression(LinearRegression):
    def __init__(self, 
                lambda_l2, 
                solver='analytical', 
                polynomial_degree=1, 
                n_iterations=1000, 
                learning_rate=0.0001):

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
                learning_rate=0.0001):

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
                learning_rate=0.0001):

        lambda_l1 = alpha * l1_ratio
        lambda_l2 = alpha * (1 - l1_ratio)
        
        super().__init__(solver='gradient_descent', 
                lambda_l2=lambda_l2, 
                lambda_l1=lambda_l1, 
                polynomial_degree=polynomial_degree, 
                n_iterations=n_iterations, 
                learning_rate=learning_rate)