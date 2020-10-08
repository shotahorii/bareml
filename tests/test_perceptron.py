import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron as SklPerceptron

import sys
sys.path.append('../')

from mlfs.supervised.perceptron import Perceptron
from mlfs.utils.validators import KFold

class PerceptronTest(unittest.TestCase):

    def test_sklearn_comparison(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        clf_skl = SklPerceptron()
        clf_mlfs = Perceptron(n_epoch=50, random_seed=1)

        skl_scores = []
        mlfs_scores = []

        kf = KFold()
        for train_idx, test_idx in kf.split(X,y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf_skl.fit(X_train, y_train)
            clf_mlfs.fit(X_train, y_train)

            skl_scores.append(clf_skl.score(X_test, y_test))
            mlfs_scores.append(clf_mlfs.score(X_test, y_test)['accuracy'])

        skl_score = np.array(skl_scores).mean()
        mlfs_score = np.array(mlfs_scores).mean()

        # accuracy difference from sklearn's Perceptron is less than 5%
        self.assertTrue(skl_score - mlfs_score < 0.05) 

unittest.main()