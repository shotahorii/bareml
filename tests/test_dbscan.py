import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as DBSCAN_skl

import sys
sys.path.append('./')
sys.path.append('../')

from bareml.utils.manipulators import StandardScaler
from bareml.unsupervised import DBSCAN

def test_dbscan():

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    X = StandardScaler().fit(X).transform(X)

    db_skl = DBSCAN_skl(eps=0.3, min_samples=10).fit(X)
    db = DBSCAN(eps=0.3, minpts=10).fit(X)

    # should be the same result for simple data like iris
    assert all([a == b for a, b in zip(db.labels_.astype(int), db_skl.labels_)])