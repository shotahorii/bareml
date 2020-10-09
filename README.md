[![PyPI version](https://badge.fury.io/py/machinelfs.svg)](https://badge.fury.io/py/machinelfs)
[![Build Status](https://travis-ci.org/shotahorii/ml-from-scratch.svg?branch=master)](https://travis-ci.org/shotahorii/ml-from-scratch)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/machinelfs)


# Machine Learning From Scratch

**machinelfs** is a Python module containing various machine learning algorithms implemented from scratch using NumPy.

The implementations are not (and not intended to be) optimised w.r.t. efficiency nor performance. Instead, they are aimed at being as straightforward/transparent as possible. 

## Installation 
```
$ pip install machinelfs
```

## List of implementations 

### Supervised Learning
- [Bernoulli Naive Bayes](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/naive_bayes.py)
- [Decision Trees](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/decision_trees.py)
- [Elastic Net](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/linear_regression.py)
- [Gaussian Naive Bayes](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/naive_bayes.py)
- [Generalised Linear Model](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/glm.py)
- [K Nearest Neighbors](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/knn.py)
- [Kernel Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/kernel_regression.py)
- [Lasso Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/linear_regression.py)
- [Linear Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/linear_regression.py)
- [Logistic Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/logistic_regression.py)
- [Perceptron](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/perceptron.py)
- [Poisson Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/glm.py)
- [Ridge Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/supervised/linear_regression.py)

### Unsupervised Learning
- [KMeans (KMeans++)](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/unsupervised/kmeans.py)

### Ensemble Learning 
- [AdaBoost](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/adaboost.py)
- [AdaBoost M1](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/adaboost.py)
- [AdaBoost Samme](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/adaboost.py)
- [AdaBoost RT](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/adaboost.py)
- [AdaBoost R2](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/adaboost.py)
- [Bagging](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/baggings.py)
- [Gradient Boosting](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/gradient_boosting.py)
- [Random Forest](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/baggings.py)
- [Stacking](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/stacking.py)
- Voting
- [XGBoost](https://github.com/shotahorii/ml-from-scratch/blob/master/machinelfs/ensemble/xgboost.py)