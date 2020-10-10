import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron as SklPerceptron

import sys
sys.path.append('./')
sys.path.append('../')

from bareml.supervised import Perceptron
from bareml.utils.validators import KFold

def test_perceptron():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    clf_skl = SklPerceptron()
    clf_bareml = Perceptron(n_epoch=50, seed=1)

    skl_scores = []
    bareml_scores = []

    kf = KFold()
    for train_idx, test_idx in kf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf_skl.fit(X_train, y_train)
        clf_bareml.fit(X_train, y_train)

        skl_scores.append(clf_skl.score(X_test, y_test))
        bareml_scores.append(clf_bareml.score(X_test, y_test)['accuracy'])

    skl_score = np.array(skl_scores).mean()
    bareml_score = np.array(bareml_scores).mean()

    # accuracy difference from sklearn's Perceptron is less than 5%
    assert skl_score - bareml_score < 0.05