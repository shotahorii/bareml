"""
K Means Clustering with K-means++ initialisation

References:
D. Arthur and S. Vassilvitskii (2007). k‐means++: the advantages of careful seeding. 
Proceedings of the Eighteenth Annual ACM‐SIAM Symposium on Discrete Algorithms. 1027–1035.
(http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. 424-428.
T. Hastie, R. Tibshirani and J. Friedman (2009). The Elements of Statistical Leraning. Springer. 509-511.
K.P. Murphy (2012). Machine Learning A Probabilistic Perspective. MIT Press. 354-357.
Y. Hirai (2012). はじめてのパターン認識. 森北出版. 155-156.
"""

# Author: Shota Horii <sh.sinker@gmail.com>

import math
import numpy as np
from scripts.distances import euclidean_distance

class KMeans:
    """ 
    K Means Clustering

    Parameters
    ----------
    k: int
        The number of clusters

    init: string
        The initialisation method ('kmeans++' or 'random')

    max_iterations: int
        The maximum number of iterations of centroid optimisation

    min_centroids_change: float
        The minimum change of centroilds to continue the iterations.
        If the change of centroids after a step is smaller than 
        this value, optimisation terminates.

    n_trials: int
        The number of trials.
        As the result of kmeans highly depending on the initialisation, 
        we repeat the trial for multiple times and 
        take the best centroids as the final result.

    """
    
    def __init__(self, k, init='kmeans++', 
                max_iterations=200, min_centroids_change=1e-15, n_trials=5):
        self.k = k
        self.init = init
        self.max_iterations = max_iterations
        self.min_centroids_change = min_centroids_change
        self.n_trials = n_trials
        self.centroids = None
        self.sum_distance = None

    def _initialise_centroids(self, X):
        """
        Initialise centroids

        Parameters
        ----------
        X: np.ndarray
            Data points to perform clustering.
            Num of rows (X.shape[0]) is the num of samples. 
            Num of columns (X.shape[1]) is the num of variables
            for clustering.
        
        Returns
        -------
        init_centroids: np.ndarray
            Initial positions of centroids.
            Num of rows (X.shape[0]) is the number of centroids (=self.k).
            Num of columns (X.shape[1]) is the num of variables.
        """
        if self.init == 'random':
            centroid_idx = np.random.choice(len(X), self.k, replace=False)
            init_centroids = X[centroid_idx]
        elif self.init == 'kmeans++':
            # implementation almost copied from below. 
            # https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
            init_centroids = X[np.random.choice(len(X), 1)]
            for _ in range(1,self.k):
                d2 = np.array([min([np.power(x - c,2).sum() for c in init_centroids]) for x in X])
                probs = d2/d2.sum()
                cumprobs = probs.cumsum()
                r = np.random.rand()
                for i, cumprob in enumerate(cumprobs):
                    if r < cumprob:
                        init_centroids = np.concatenate((init_centroids,X[[i]]))
                        break
        else:
            raise ValueError("Set init parameter as either 'random' or 'kmeans++'.")

        return init_centroids

    def _assign_closest_centroid(self, X, centroids):
        """
        Assigns their closest centroid to each of data point in X

        Parameters
        ----------
        X: np.ndarray
            Data points to perform clustering.
            Num of rows (X.shape[0]) is the num of samples.
            Num of columns (X.shape[1]) is the num of variables
            for clustering.

        centroids: np.ndarray
            Positions of centroids.
            Num of rows (X.shape[0]) is the number of centroids (=self.k).
            Num of columns (X.shape[1]) is the num of variables.
        
        Returns
        -------
        assigned_centroid: np.ndarray (1d array)
            List of assigned centroids to each data point in X.
            Num of elements (assigned_centroid.shape[0]) 
            is the num of samples.
            Value in the array indicates the index of centroid 
            which is assigned to the data point.

        dist_to_assigned_centroid: np.ndarray (1d array)
            List of distances between each data point in X and 
            its closest centroid (assigned centroid).
            Num of elements (dist_to_assigned_centroid.shape[0]) 
            is the num of samples.
            Value in the array indicates the distance. 
        """
        assigned_centroid = np.zeros(len(X))
        dist_to_assigned_centroid = np.zeros(len(X))

        for i, x in enumerate(X):
            closest_distance = np.inf
            closest_centroid_idx = 0

            for j, c in enumerate(centroids):
                d = euclidean_distance(x,c)
                if d < closest_distance:
                    closest_distance = d
                    closest_centroid_idx = j

            assigned_centroid[i] = closest_centroid_idx
            dist_to_assigned_centroid[i] = closest_distance
        
        return assigned_centroid, dist_to_assigned_centroid

    def fit(self, X):
        """
        Compute K (=self.k) centroids from the given iput data X

        Parameters
        ----------
        X: np.ndarray
            Data points to perform clustering.
            Num of rows (X.shape[0]) is the num of samples.
            Num of columns (X.shape[1]) is the num of variables
            for clustering.
        
        Returns
        -------
        self: KMeans
        """
        
        best_centroids = None
        min_sum_distance = np.inf

        for trial in range(self.n_trials):

            # initialise
            curr_centroids = self._initialise_centroids(X)

            for _ in range(self.max_iterations):
                
                prev_centroids = np.copy(curr_centroids)

                assigned_centroid, dist_to_assigned_centroid = self._assign_closest_centroid(X, curr_centroids)

                for i in range(self.k):
                    curr_centroids[i] = X[assigned_centroid==i].mean(axis=0)

                centroids_change = max(np.sqrt(np.power(curr_centroids - prev_centroids, 2).sum(axis=1)))
                if centroids_change < self.min_centroids_change:
                    break 
            
            assigned_centroid, dist_to_assigned_centroid = self._assign_closest_centroid(X, curr_centroids)
            sum_distance = dist_to_assigned_centroid.sum()

            if sum_distance < min_sum_distance:
                min_sum_distance = sum_distance
                best_centroids = np.copy(curr_centroids)

        self.centroids = best_centroids
        self.sum_distance = min_sum_distance

        return self

    def predict(self, X):
        """
        Assign the closest centroid (cluster) to each data point in X

        Parameters
        ----------
        X: np.ndarray
            Data points to compute closest centroid (= assign a cluster)
            Num of rows (X.shape[0]) is the num of samples.
            Num of columns (X.shape[1]) is the num of variables.
        
        Returns
        -------
        assigned_centroid: np.ndarray (1d array)
            List of assigned centroids to each data point in X.
            Num of elements (assigned_centroid.shape[0]) 
            is the num of samples.
            Value in the array indicates the index of centroid 
            which is assigned to the data point.
        """
        assigned_centroid, dist_to_assigned_centroid = self._assign_closest_centroid(X, self.centroids)
        return assigned_centroid
