[![Build Status](https://travis-ci.org/shotahorii/bareml.svg?branch=master)](https://travis-ci.org/shotahorii/bareml)
[![PyPI version](https://badge.fury.io/py/bareml.svg)](https://badge.fury.io/py/bareml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bareml)

![Logo](/logo.png)

**bareml** is a Python module containing various machine learning algorithms implemented from scratch using NumPy.

The implementations are not (and not intended to be) optimised w.r.t. efficiency nor performance. Instead, they are aimed at being as straightforward/transparent as possible. 

## Installation 
```
$ pip install bareml
```

## List of implementations 

### Supervised Learning
- [Bernoulli Naive Bayes](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/naive_bayes.py)
- [Decision Trees](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/decision_trees.py)
- [Elastic Net](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/linear_regression.py)
- [Gaussian Naive Bayes](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/naive_bayes.py)
- [Generalised Linear Model](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/glm.py)
- [K Nearest Neighbors](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/knn.py)
- [Kernel Ridge Regression](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/kernel_regression.py)
- [Lasso Regression](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/linear_regression.py)
- [Linear Regression](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/linear_regression.py)
- [Logistic Regression](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/logistic_regression.py)
- [Perceptron](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/perceptron.py)
- [Poisson Regression](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/glm.py)
- [Ridge Regression](https://github.com/shotahorii/bareml/blob/master/bareml/supervised/linear_regression.py)

### Unsupervised Learning
- [KMeans (KMeans++)](https://github.com/shotahorii/bareml/blob/master/bareml/unsupervised/kmeans.py)
- [DBSCAN](https://github.com/shotahorii/bareml/blob/master/bareml/unsupervised/dbscan.py)

### Ensemble Learning 
- [AdaBoost](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/adaboost.py)
- [AdaBoost M1](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/adaboost.py)
- [AdaBoost Samme](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/adaboost.py)
- [AdaBoost RT](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/adaboost.py)
- [AdaBoost R2](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/adaboost.py)
- [Bagging](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/baggings.py)
- [Gradient Boosting](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/gradient_boosting.py)
- [Random Forest](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/baggings.py)
- [Stacking](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/stacking.py)
- Voting
- [XGBoost](https://github.com/shotahorii/bareml/blob/master/bareml/ensemble/xgboost.py)

### Utilities
- [Data manipulators (Scaler, Encoder etc)](https://github.com/shotahorii/bareml/blob/master/bareml/utils/manipulators.py)
- [Distance metrics](https://github.com/shotahorii/bareml/blob/master/bareml/utils/distances.py)
- [Evaluation Metrics](https://github.com/shotahorii/bareml/blob/master/bareml/utils/metrics.py)
- [Kernel functions](https://github.com/shotahorii/bareml/blob/master/bareml/utils/kernels.py)
- [Probability Distributions](https://github.com/shotahorii/bareml/blob/master/bareml/utils/probability_distribution.py)
- [Tools for validation (KFold, StratifiedKFold etc)](https://github.com/shotahorii/bareml/blob/master/bareml/utils/validators.py)