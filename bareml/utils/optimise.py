"""
Optimisers

Author: Shota Horii <sh.sinker@gmail.com>

References:
https://www.coursera.org/lecture/ml-regression/how-to-handle-the-intercept-3KZiN

"""

from abc import ABC, abstractmethod
import math
import numpy as np

from .misc import supremum_eigen
from .activation_functions import Sigmoid, Softmax


#########################
# Weight Initialisation #
#########################


def initialise_random(n, m=None):
    """
    Initialise (n * m) shaped weights randomly.

    Parameters
    ----------
    n, m: int
        shape of the weight to initialise. (n * m)
        if m is none, weight is a 1d array of length = n

    Returns
    -------
    w: np.ndarray (n, m) float
    """
    # initialise weights in [-1/sqrt(n), 1/sqrt(n)) 
    # why? see below articles. 
    # https://leimao.github.io/blog/Weights-Initialization/
    # https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    limit = 1 / math.sqrt(n) if m is None else 1 / math.sqrt(n * m)
    size = n if m is None else (n, m)

    return np.random.uniform(-limit, limit, size)


def initialise_zero(n, m=None):
    """
    Initialise (n * m) shaped weights to zero. 

    Parameters
    ----------
    n, m: int
        shape of the weight to initialise. (n * m)
        if m is none, weight is a 1d array of length = n

    Returns
    -------
    w: np.ndarray (n, m) float
    """
    size = n if m is None else (n, m)
    return np.zeros(size)


###############################
# Hyperparameter Optimisation #
###############################


def auto_lr(X):
    """
    Find good learning rate for square error minimisation. 
    opt learning rate = 1/p where p is max eigen value of X.T @ X
    Use Gershgorin circle theorem to estimate supremum of the 
    eiven vectors, instead of calculating actual max eigen value. 
    Ref(in JP): https://qiita.com/fujiisoup/items/e7f703fc57e2dfc441ad

    Parameters
    ----------
    X: np.array (n,d)
        feature matrix
        n: number of samples
        d: number of features
    """
    rho = supremum_eigen(X.T @ X)
    return 1.0 / rho


##############
# Optimisers #
##############


class Optimiser(ABC):

    @abstractmethod
    def solve(self):
        pass
    

class GradientDescent(Optimiser):
    """
    Gradient Descent

    Parameters
    ----------
    gradient: a function to calculate gradient of loss function w.r.t. w
        parameters of the function must be (X, y, w)
        return of the function must be a (d,) array representing gradient

    max_iter: int > 0
        max number of iterations
    
    tol: float >= 0
        conversion criterion. if delta of w is smaller than tol, 
        algorighm considers it's converged.

    lr: float > 0
    """

    def __init__(self, gradient, max_iter=1000, tol=1e-4, lr=None):
        self.gradient = gradient
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

    def solve(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray (n,d) 
            n: number of samples
            d: number of features
        
        y: np.ndarray (n,) or (n,c)
            n: number of samples
            c: number of classes
        """

        # if learning rate is not given, set automatically
        lr = self.lr if self.lr else auto_lr(X)
        
        # initialise the weights as zero
        if y.ndim == 1:
            w = initialise_zero(X.shape[1])
        else: # y.ndim == 2 i.e. classification
            w = initialise_zero(X.shape[1], y.shape[1])
        
        for _ in range(self.max_iter):
            # step
            grads = self.gradient(X, y, w)
            w_new = w - lr * grads
            # check the conversion
            if (np.abs(w_new - w) < self.tol).all():
                print('Converged.')
                return w_new
            # update
            w = w_new
        
        print('Not converged.')
        return w


class LassoISTA(Optimiser):
    """
    Lasso Optimiser with ISTA (Iterative Shrinkage Thresholding Algorithm). 
    Ref (in JP): https://qiita.com/fujiisoup/items/f2fe3b508763b0cc6832

    Parameters
    ----------
    alpha: float > 0
        l1 reguralisation parameter
    
    max_iterations: int > 0
        max number of iterations
    
    tol: float >= 0
        conversion criterion. if delta of w is smaller than tol, 
        algorighm considers it's converged.
    """
    
    def __init__(self, alpha=0, max_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tol = tol

    def solve(self, X, y, has_intercept=True):
        """
        Solve Lasso.
        argmin_w ( 1/2 (y - Xw)^2 + alpha |w| )

        As this cannot be solved analytically, we solve this in iterative way. 
        In each step t, (call w of t-th step as w_t here),  we approximate
        L(w) = 1/2 (y-Xw)^2 with an another function G(w; w_t)
        so that we can solve argmin_w ( G(w; w_t) + alpha |w| ) in each step. 

        G(w; w_t) = L(w_t) + dL(w_t)/dw (w - w_t) + rho/2 |w - w_t|^2
                = rho/2 |(w_t - 1/rho dL(w_t)/dw) - w|^2 + const 
        where rho is the maximum eigen value of the square matrix X.T @ X

        Note that, always L(w) <= G(w; w_t) and L(w_t) == G(w_t; w_t)
        Hence, moving towards the direction of minimising G(w; w_t)
        is guaranteed to be the direction of minimising L(w) as well. 

        With G(w; w_t) above, we can solve argmin_w ( G(w; w_t) + alpha |w| )
        using the soft threshold function. 
        So each step, we can move on to optimise G(w; w_t), which is also 
        moving towards minimum of ( 1/2 (y - Xw)^2 + alpha |w| ).

        Parameters
        ----------
        X: np.ndarray (n, d) 
            predictor variables matrix
            n: number of samples
            d: number of features
        
        y: np.ndarray (n,)
            target variables
            n: number of samples

        has_intercept: bool
            if X contains intercept (1st column filled with all 1.)
        """
        n_samples, n_features = X.shape
        
        # initialise w_t (w of t-th step)
        w_t = initialise_zero(n_features)
        # rho is the maximum eigen value of the square matrix X.T @ X
        rho = supremum_eigen(X.T @ X)
        # threshold of soft threshold function. weighted with num of samples.
        threshold = n_samples * self.alpha / rho

        for _ in range(self.max_iterations):
            
            # in each step t, with w_t, solve argmin_w (G(w; w_t) + alpha |w|)
            # G(w; w_t) = rho/2 |(w_t - 1/rho dL(w_t)/dw) - w|^2 + const 
            # L(w) = 1/2 (y - Xw)^2  =>  dL/dw = -X^T (y - Xw)
            dl_dw = -X.T @ (y - X @ w_t) 
            w_new = self._soft_threashold(w_t - dl_dw/rho, threshold)

            if has_intercept:
                # exclude intercept from reguralisation 
                # by setting threshold 0. 
                w_zero_threshold = self._soft_threashold(w_t - dl_dw/rho, 0)
                w_new[0] = w_zero_threshold[0]

            if (np.abs(w_new - w_t) < self.tol).all():
                print('Converged.')
                return w_new
            w_t = w_new
        
        print('Not converged.')
        return w_t

    def _soft_threashold(self, lw, threshold):
        """
        Soft threshold function

        Parameters
        ----------
        lw: np.ndarray (d,)
            d: number of features
            Array of values of w, which minimise L(w) without alpha*|w| factor

        threshold: float > 0
            the threshold of the soft threshold function

        Returns
        -------
        np.ndarray (d,)    
            if lw > threshold, return lw - threshold
            if lw < -threshold, return lw + threshold
            if -threshold <= lw <= threshold, reutrn 0
        """
        return np.sign(lw) * np.maximum(np.abs(lw) - threshold, 0.0)


