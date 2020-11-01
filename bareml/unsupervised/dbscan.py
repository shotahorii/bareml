"""
DBSCAN clustering

Author: Shota Horii <sh.sinker@gmail.com>
Test: tests/test_dbscan.py

References:
https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
"""

import numpy as np

from ..base import Clustering
from ..utils.distances import euclidean_distance


class DBSCAN(Clustering):
    """
    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Direct implementation of algorism on the original paper.

    Parameters
    ----------
    eps: float > 0
        The maximum distance between two samples for one 
        to be considered as in the neighborhood of the other.

    minpts: int > 0
        The number of samples in a neighborhood for a point 
        to be considered as a core point. This includes the point itself.

    distance: function (default is euclidean_distance)
        distance metrics to be used to calculate distance between 2 points.
    """

    def __init__(self, eps, minpts, distance=euclidean_distance):
        self.eps = eps
        self.minpts = minpts
        self.dist = distance

        # cluster labels
        self.labels_ = None

    def _predict(self, X):
        raise NotImplementedError('DBSCAN does only support fit or fit_predict.')

    def _fit(self, X):
        """
        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features
        
        Returns
        -------
        self
        """

        # initialise cluster id as 0
        cl_id = 0
        # initialise cluster labels with all NaN
        self.labels_ = np.full(len(X),np.nan)

        for p_idx in range(len(X)):
            if np.isnan(self.labels_[p_idx]):
                if self._expand_cluster(X, p_idx, cl_id):
                    cl_id += 1
        
        return self

    def _fit_predict(self, X):
        """
        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features
        
        Returns
        -------
        labels_: np.ndarray (n,) of int
            n: number of samples
            value indicates each cluster id. -1 means noise. 
        """

        # initialise cluster id as 0
        cl_id = 0
        # initialise cluster labels with all NaN
        self.labels_ = np.full(len(X),np.nan)

        for p_idx in range(len(X)):
            if np.isnan(self.labels_[p_idx]):
                if self._expand_cluster(X, p_idx, cl_id):
                    cl_id += 1
        
        return self.labels_

    def _expand_cluster(self, X, p_idx, cl_id):
        """
        Core calculation

        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features

        p_idx: int >= 0
            index of the point in focus

        Returns
        -------
        if_update_cl_id: bool
            whether or not to update cl_id
        """

        NOISE_ID = -1

        seeds = self._region_query(X, p_idx)

        if len(seeds) < self.minpts:
            self.labels_[p_idx] = NOISE_ID
            return False
        
        self.labels_[list(seeds)] = cl_id

        # remove the point p itself
        seeds.remove(p_idx)

        while len(seeds) > 0:
            current_idx = seeds.pop()
            result = self._region_query(X, current_idx)

            if len(result) >= self.minpts:
                for result_idx in result:
                    if np.isnan(self.labels_[result_idx]) or self.labels_[result_idx] == NOISE_ID:
                        if np.isnan(self.labels_[result_idx]):
                            seeds.add(result_idx)
                        self.labels_[result_idx] = cl_id
        return True

    def _region_query(self, X, p_idx):
        """
        Get the list of neighbour points around pi in X.

        Parameters
        ----------
        X: np.ndarray (n,d) of real (-inf, inf)
            n: number of samples
            d: number of features

        p_idx: int >= 0
            index of the point to check the neighbour

        Returns
        -------
        neighbours: set
            set of idx of points around p including p itself
        """
        p = X[p_idx]
        # distance between p and each point q in X
        distances = np.apply_along_axis(lambda q: self.dist(p,q), axis=1, arr=X)
        # list of idx of points around p (including p_idx itself)
        neighbours = np.where(distances <= self.eps)[0]
        
        return set(neighbours)

