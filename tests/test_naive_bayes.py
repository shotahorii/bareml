import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB as GaussianNB_skl
from sklearn.naive_bayes import BernoulliNB as BernoulliNB_skl

import sys
sys.path.append('./')
sys.path.append('../')

from bareml.supervised import GaussianNB, BernoulliNB
from bareml.utils.validators import train_test_split

def test_gaussian():
    data = load_iris()
    X = data.data
    y = data.target

    clf_skl = GaussianNB_skl()
    clf_bareml = GaussianNB()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

    clf_skl.fit(X_train, y_train)
    clf_bareml.fit(X_train, y_train)

    preds_skl = clf_skl.predict(X_test).tolist()
    preds_bareml = clf_bareml.predict(X_test).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_gaussian_onehot():
    data = load_iris()
    X = data.data
    y = data.target

    clf_skl = GaussianNB_skl()
    clf_bareml = GaussianNB()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)
    
    # one-hot encoding for bareml classifier
    y_train_onehot = pd.get_dummies(y_train).values

    clf_skl.fit(X_train, y_train)
    clf_bareml.fit(X_train, y_train_onehot)

    preds_skl = clf_skl.predict(X_test).tolist()
    preds_bareml = np.argmax(clf_bareml.predict(X_test),axis=1)

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_bernoulli():
    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 100))
    y = np.array([0, 1, 2, 3, 3, 4])

    clf_skl = BernoulliNB_skl()
    clf_bareml = BernoulliNB()

    clf_skl.fit(X, y)
    clf_bareml.fit(X, y)

    preds_skl = clf_skl.predict(X).tolist()
    preds_bareml = clf_bareml.predict(X).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_bernoulli_onehot():
    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 100))
    y = np.array([0, 1, 2, 3, 3, 4])
    y_onehot = pd.get_dummies(y).values

    clf_skl = BernoulliNB_skl()
    clf_bareml = BernoulliNB()

    clf_skl.fit(X, y)
    clf_bareml.fit(X, y_onehot)

    preds_skl = clf_skl.predict(X).tolist()
    preds_bareml = np.argmax(clf_bareml.predict(X),axis=1)

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])