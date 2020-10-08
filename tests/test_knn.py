import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import sys
sys.path.append('../')

from mlfs.supervised.knn import KNNClassifier, KNNRegressor
from mlfs.utils.validators import train_test_split

class KNNTest(unittest.TestCase):

    def test_regressor(self):
        data = load_boston()
        X = data.data
        y = data.target

        reg_skl = KNeighborsRegressor(3) #k=3
        reg_mlfs = KNNRegressor(3) #k=3

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

        reg_skl.fit(X_train,y_train)
        reg_mlfs.fit(X_train,y_train)

        preds_skl = reg_skl.predict(X_test).tolist()
        preds_mlfs = reg_mlfs.predict(X_test).tolist()

        # should be the same result
        self.assertListEqual(preds_skl, preds_mlfs)

    def test_classifier_multi(self):
        data = load_iris()
        X = data.data
        y = data.target

        clf_skl = KNeighborsClassifier(3) #k=3
        clf_mlfs = KNNClassifier(3) #k=3

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)
        
        # one-hot encoding for mlfs classifier
        y_train_onehot = pd.get_dummies(y_train).values

        clf_skl.fit(X_train,y_train)
        clf_mlfs.fit(X_train,y_train_onehot)

        preds_skl = clf_skl.predict(X_test).tolist()
        # decoding one-hot 
        preds_mlfs = np.argmax(clf_mlfs.predict(X_test),axis=1).tolist()

        # should be the same result
        self.assertListEqual(preds_skl, preds_mlfs)

    def test_classifier_binary(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        clf_skl = KNeighborsClassifier(3) #k=3
        clf_mlfs = KNNClassifier(3) #k=3

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

        clf_skl.fit(X_train,y_train)
        clf_mlfs.fit(X_train,y_train)

        preds_skl = clf_skl.predict(X_test).tolist()
        preds_mlfs = clf_mlfs.predict(X_test).tolist()

        # should be the same result
        self.assertListEqual(preds_skl, preds_mlfs)

unittest.main()