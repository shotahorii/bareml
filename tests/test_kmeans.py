import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans as SklKMeans

import sys
sys.path.append('./')
sys.path.append('../')

from bareml.unsupervised import KMeans

def test_kmeans():
    data = load_iris()
    X = data.data

    cls_skl = SklKMeans(3)
    cls_bareml = KMeans(3, n_trials=10)

    cls_skl.fit(X)
    cls_bareml.fit(X)

    res_skl = cls_skl.predict(X)
    res_bareml = cls_bareml.predict(X)

    # match labeling of bareml and sklearn
    res_bareml += 1000 # assume k < 1000
    labels_bareml = []
    for e in res_bareml:
        if e not in labels_bareml: 
            labels_bareml.append(e)
            
    for i,e in enumerate(labels_bareml):
        res_bareml[res_bareml==e] = i

    res_skl += 1000 # assume k < 1000
    labels_skl = []
    for e in res_skl:
        if e not in labels_skl: 
            labels_skl.append(e)
            
    for i,e in enumerate(labels_skl):
        res_skl[res_skl==e] = i

    # should be the same result for simple data like iris
    assert all([a == b for a, b in zip(res_bareml, res_skl)])