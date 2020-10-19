import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression as SklLinearRegression
from sklearn.linear_model import Ridge

import sys
sys.path.append('./')
sys.path.append('../')

from bareml.supervised import LinearRegression, RidgeRegression
from bareml.utils.validators import train_test_split

def test_analytical():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = SklLinearRegression()
    reg_bareml = LinearRegression(solver='analytical')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    # round in 4 decimals
    preds_skl = np.round(reg_skl.predict(X_test),4).tolist() 
    preds_bareml = np.round(reg_bareml.predict(X_test),4).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_gradient_descent():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = SklLinearRegression()
    reg_bareml = LinearRegression(solver='GD')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    preds_skl = reg_skl.predict(X_test)
    preds_bareml = reg_bareml.predict(X_test)

    # close enough
    assert all(np.abs(preds_skl - preds_bareml)<0.05)


def test_ridge_analytical():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = Ridge(alpha=1.0)
    reg_bareml = RidgeRegression(alpha=1.0,solver='analytical')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    # round in 4 decimals
    preds_skl = np.round(reg_skl.predict(X_test),4).tolist() 
    preds_bareml = np.round(reg_bareml.predict(X_test),4).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_ridge_gradient_descent():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = Ridge(alpha=1.0)
    reg_bareml = RidgeRegression(alpha=1.0,solver='GD')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    preds_skl = reg_skl.predict(X_test)
    preds_bareml = reg_bareml.predict(X_test)

    # close enough
    assert all(np.abs(preds_skl - preds_bareml)<2.0)