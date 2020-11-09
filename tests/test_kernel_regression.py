import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.kernel_ridge import KernelRidge as SklKernelRidge

import sys
sys.path.append('../')

from bareml.machinelearning.supervised import KernelRidge
from bareml.machinelearning.utils.model_selection import train_test_split


def test_linear_kernel():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = SklKernelRidge(alpha=1.0)
    reg_bareml = KernelRidge(alpha=1.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    # round in 4 decimals
    preds_skl = np.round(reg_skl.predict(X_test),4).tolist() 
    preds_bareml = np.round(reg_bareml.predict(X_test),4).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_rbf_kernel():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = SklKernelRidge(alpha=1.0,kernel='rbf')
    reg_bareml = KernelRidge(alpha=1.0,kernel='rbf')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    # round in 4 decimals
    preds_skl = np.round(reg_skl.predict(X_test),4).tolist() 
    preds_bareml = np.round(reg_bareml.predict(X_test),4).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_sigmoid_kernel():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = SklKernelRidge(alpha=1.0,kernel='sigmoid')
    reg_bareml = KernelRidge(alpha=1.0,kernel='sigmoid')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    # round in 4 decimals
    preds_skl = np.round(reg_skl.predict(X_test),4).tolist() 
    preds_bareml = np.round(reg_bareml.predict(X_test),4).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])


def test_polynomial_kernel():
    data = load_boston()
    X = data.data
    y = data.target

    reg_skl = SklKernelRidge(alpha=2.0,kernel='polynomial',degree=2)
    reg_bareml = KernelRidge(alpha=2.0,kernel='polynomial',degree=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    reg_skl.fit(X_train,y_train)
    reg_bareml.fit(X_train,y_train)

    # with polynomial kernel, sklearn's result is not quite stable
    # so check with only 1 decimal
    preds_skl = np.round(reg_skl.predict(X_test),1).tolist() 
    preds_bareml = np.round(reg_bareml.predict(X_test),1).tolist()

    # should be the same result
    assert all([a == b for a, b in zip(preds_bareml, preds_skl)])