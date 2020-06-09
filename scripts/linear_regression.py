"""
Linear Regression (Ridge, Lasso and ElasticNet)

References:
<Lasso>
https://qiita.com/fujiisoup/items/f2fe3b508763b0cc6832 (in JP)

<Elastic Net>
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 458, 477.

"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np

from scripts.preprocessing import polynomial_features, StandardScaler
from scripts.solvers import PInv, LassoISTA, LeastSquareGD
from scripts.metrics import mean_square_error

class LinearRegression:
    """
    Linear Regression Model. 
    Base class for Ridge Regression, Lasso Regression and ElasticNet Regression.
    
    Parameters
    ----------
    solver: string {'pinv','gradient_descent','lasso'}
        Solver to be used in fit function.

    alpha_l1: float >=0
        L1 reguralisation parameter.
    
    alpha_l2: float >= 0
        L2 reguralisation parameter.

    polynomial_degree: int {1,2,3,...}
        If more than 1, generate polynomial and interaction features from 
        input predictor variable X, with the given degree.

    max_iterations: int > 0
        Maximum number of iterations. Applicable to solvers using iterative method. 
    
    tol: float >= 0
        Conversion criterion. Applicable to solvers using iterative method.
        In each step, if delta is smaller than tol, algorighm considers it's converged.
    
    learning_rate: float in (0,1)
        Learning rate parameter for Gradient Descent algorithm.
    """
    
    def __init__(self, solver='pinv', alpha_l1=0, alpha_l2=0, polynomial_degree=1, 
        max_iterations=1000, tol=1e-4, learning_rate=None):
        
        self.solver = solver
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.polynomial_degree = polynomial_degree
        self.max_iterations = max_iterations
        self.tol = tol
        self.learning_rate = learning_rate
        self.w = None
        self.train_error = None

        # GD and LassoISTA solver need feature scaling 
        self.scaler = StandardScaler() if solver in ['gradient_descent', 'lasso'] else None


    def fit(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray (n, d) 
            predictor variables matrix
            n: number of samples
            d: number of features
        
        y: np.ndarray (n,)
            target variables
            n: number of samples

        Returns
        -------
        self: LinearRegression
        """

        X = polynomial_features(X, self.polynomial_degree)

        if self.scaler:
            # exclude bias (all 1) from scaling 
            X[:,1:] = self.scaler.fit(X[:,1:]).transform(X[:,1:])

        if self.solver == 'pinv':
            pinv = PInv(alpha=self.alpha_l2)
            self.w = pinv.solve(X, y)

        elif self.solver == 'lasso':
            if self.alpha_l2==0: # Lasso regression
                lasso = LassoISTA(self.alpha_l1, self.max_iterations, self.tol)
                self.w = lasso.solve(X, y)
            else: # Elastic Net
                # we solve elastic net as lasso. (Murphy 2012)
                d = X.shape[1] # num features 
                c = np.power(1+self.alpha_l2, -0.5)
                X_concat = math.sqrt(self.alpha_l2) * np.eye(d)
                y_concat = np.zeros(d)

                X_new = c * np.concatenate([X, X_concat], axis=0)
                y_new = np.concatenate([y, y_concat], axis=0)

                elastic = LassoISTA(c * self.alpha_l1, self.max_iterations, self.tol)
                self.w = c * elastic.solve(X_new, y_new)

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
        """
        Parameters
        ----------
        X: np.ndarray (n, d) 
            predictor variables matrix
            n: number of samples
            d: number of features

        Returns
        -------
        y_pred: np.ndarray (n,)
            predicted target variables
            n: number of samples
        """

        X = polynomial_features(X, self.polynomial_degree)

        if self.scaler:
            # exclude bias (all 1) from scaling 
            X[:,1:] = self.scaler.transform(X[:,1:])

        return np.dot(self.w, X.T)


class RidgeRegression(LinearRegression):
    """
    Ridge Regression Model.
    
    Parameters
    ----------
    solver: string {'pinv','gradient_descent'}
        Solver to be used in fit function.
    
    alpha: float >= 0
        L2 reguralisation parameter.

    polynomial_degree: int {1,2,3,...}
        If more than 1, generate polynomial and interaction features from 
        input predictor variable X, with the given degree.

    max_iterations: int > 0
        Maximum number of iterations. Applicable to solvers using iterative method. 
    
    tol: float >= 0
        Conversion criterion. Applicable to solvers using iterative method.
        In each step, if delta is smaller than tol, algorighm considers it's converged.
    
    learning_rate: float in (0,1)
        Learning rate parameter for Gradient Descent algorithm.
    """
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
    """
    Lasso Regression Model.
    
    Parameters
    ----------
    alpha: float >=0
        L1 reguralisation parameter.

    polynomial_degree: int {1,2,3,...}
        If more than 1, generate polynomial and interaction features from 
        input predictor variable X, with the given degree.

    max_iterations: int > 0
        Maximum number of iterations. Applicable to solvers using iterative method. 
    
    tol: float >= 0
        Conversion criterion. Applicable to solvers using iterative method.
        In each step, if delta is smaller than tol, algorighm considers it's converged.
    """
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
    """
    ElasticNet Regression Model.
    
    Parameters
    ----------
    alpha: float >=0
        Reguralisation parameter distributed to L1 and L2 penalty factors
    
    l1_ratio: float in [0,1]
        Parameter to define distribution ratio of alpha between L1 and L2 
        penalty factors. If l1_ratio=1, it's Lasso Regression.
        If l1_ratio=0, it's Ridge Regression.

    polynomial_degree: int {1,2,3,...}
        If more than 1, generate polynomial and interaction features from 
        input predictor variable X, with the given degree.

    max_iterations: int > 0
        Maximum number of iterations. Applicable to solvers using iterative method. 
    
    tol: float >= 0
        Conversion criterion. Applicable to solvers using iterative method.
        In each step, if delta is smaller than tol, algorighm considers it's converged.
    """
    def __init__(self, 
                alpha,
                l1_ratio,
                polynomial_degree=1, 
                max_iterations=1000, 
                tol=1e-4):

        alpha_l1 = alpha * l1_ratio
        alpha_l2 = alpha * (1 - l1_ratio)
        
        super().__init__(solver='lasso', 
                alpha_l1=alpha_l1, 
                alpha_l2=alpha_l2, 
                polynomial_degree=polynomial_degree, 
                max_iterations=max_iterations, 
                tol=tol)