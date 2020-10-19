import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression as LogisticRegression_skl

import sys
sys.path.append('./')
sys.path.append('../')

from bareml.supervised import LogisticRegression
from bareml.utils.validators import KFold, StratifiedKFold


def test_binary_classification():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    clf_skl = LogisticRegression_skl()
    clf_bareml = LogisticRegression()

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

    # accuracy difference from sklearn's LogisticRegression is less than 5%
    assert skl_score - bareml_score < 0.05


def test_multi_classification():
    data = load_iris()
    X = data.data
    y = data.target

    clf_skl = LogisticRegression_skl()
    clf_bareml = LogisticRegression()

    skl_scores = []
    bareml_scores = []

    kf = StratifiedKFold()
    for train_idx, test_idx in kf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf_skl.fit(X_train, y_train)
        clf_bareml.fit(X_train, y_train)

        skl_scores.append(clf_skl.score(X_test, y_test))
        bareml_scores.append(clf_bareml.score(X_test, y_test)['accuracy'])

    skl_score = np.array(skl_scores).mean()
    bareml_score = np.array(bareml_scores).mean()

    # accuracy difference from sklearn's LogisticRegression is less than 5%
    assert skl_score - bareml_score < 0.05


def test_multi_classification_onehot():
    data = load_iris()
    X = data.data
    y = data.target

    clf_skl = LogisticRegression_skl()
    clf_bareml = LogisticRegression()

    skl_scores = []
    bareml_scores = []

    kf = StratifiedKFold()
    for train_idx, test_idx in kf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        y_train_onehot = pd.get_dummies(y_train).values
        y_test_onehot = pd.get_dummies(y_test).values

        clf_skl.fit(X_train, y_train)
        clf_bareml.fit(X_train, y_train_onehot)

        skl_scores.append(clf_skl.score(X_test, y_test))
        bareml_scores.append(clf_bareml.score(X_test, y_test_onehot)['accuracy'])

    skl_score = np.array(skl_scores).mean()
    bareml_score = np.array(bareml_scores).mean()

    # accuracy difference from sklearn's LogisticRegression is less than 5%
    assert skl_score - bareml_score < 0.05
