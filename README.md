# Machine Learning From Scratch

**mlfs** is a Python module containing various machine learning algorithms, which are implemented from scratch using NumPy.

The implementations are not (and not intended to be) optimised w.r.t. efficiency nor performance. Instead, they are aimed at being as straightforward/transparent as possible. 

This module doesn't include deep learning. For deep learning, see [dlfs](https://github.com/shotahorii/dl-from-scratch).

## Installation 
```
pip install mlfs
```

## List of implementations 

### Supervised Learning
- [Perceptron](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/perceptron.py)
- [K Nearest Neighbors](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/knn.py)
- [Gaussian Naive Bayes](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/naive_bayes.py)
- [Bernoulli Naive Bayes](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/naive_bayes.py)
- [Logistic Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/logistic_regression.py)
- [Linear Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/linear_regression.py)
- [Ridge Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/linear_regression.py)
- [Lasso Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/linear_regression.py)
- [Elastic Net](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/linear_regression.py)
- [Kernel Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/kernel_regression.py)
- [Generalised Linear Model](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/glm.py)
- [Poisson Regression](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/glm.py)
- [Decision Trees](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/supervised/decision_trees.py)

### Unsupervised Learning
- [KMeans (KMeans++)](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/unsupervised/kmeans.py)

### Ensemble Learning 
- Voting
- [Stacking](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/stacking.py)
- [Bagging](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/baggings.py)
- [Random Forest](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/baggings.py)
- [Gradient Boosting](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/gradient_boosting.py)
- [XGBoost](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/xgboost.py)
- [AdaBoost](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/adaboost.py)
- [AdaBoost M1](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/adaboost.py)
- [AdaBoost Samme](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/adaboost.py)
- [AdaBoost RT](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/adaboost.py)
- [AdaBoost R2](https://github.com/shotahorii/ml-from-scratch/blob/master/mlfs/ensemble/adaboost.py)

