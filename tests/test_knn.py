import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import sys
sys.path.append('./')
sys.path.append('../')

from machinelfs.supervised.knn import KNNClassifier, KNNRegressor
from machinelfs.utils.validators import train_test_split

def test_regressor():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = KNeighborsRegressor(3) #k=3
    reg_machinelfs = KNNRegressor(3) #k=3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

    reg_skl.fit(X_train,y_train)
    reg_machinelfs.fit(X_train,y_train)

    preds_skl = reg_skl.predict(X_test).tolist()
    preds_machinelfs = reg_machinelfs.predict(X_test).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_machinelfs, preds_skl)])


def test_classifier_multi():
    data = load_iris()
    X = data.data
    y = data.target

    clf_skl = KNeighborsClassifier(3) #k=3
    clf_machinelfs = KNNClassifier(3) #k=3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)
    
    # one-hot encoding for machinelfs classifier
    y_train_onehot = pd.get_dummies(y_train).values

    clf_skl.fit(X_train,y_train)
    clf_machinelfs.fit(X_train,y_train_onehot)

    preds_skl = clf_skl.predict(X_test).tolist()
    # decoding one-hot 
    preds_machinelfs = np.argmax(clf_machinelfs.predict(X_test),axis=1).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_machinelfs, preds_skl)])


def test_classifier_binary():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    clf_skl = KNeighborsClassifier(3) #k=3
    clf_machinelfs = KNNClassifier(3) #k=3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

    clf_skl.fit(X_train,y_train)
    clf_machinelfs.fit(X_train,y_train)

    preds_skl = clf_skl.predict(X_test).tolist()
    preds_machinelfs = clf_machinelfs.predict(X_test).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_machinelfs, preds_skl)])