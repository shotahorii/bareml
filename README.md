[![Build Status](https://travis-ci.org/shotahorii/bareml.svg?branch=master)](https://travis-ci.org/shotahorii/bareml)
[![PyPI version](https://badge.fury.io/py/bareml.svg)](https://badge.fury.io/py/bareml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bareml)

![Logo](/logo.png)

**bareml** is a set of "bare" implementations of machine learning / deep learning algorithms from scratch (only depending on numpy) in Python. "bare" means to aim at:
1. Code as a direct translation of the algorithm / formula
2. With minimum error handling and efficiency gain tricks

To maximise understandability of the code, interface of modules in `bareml/machinelearning/` is aligned to *Scikit-learn*, and interface of modules in `bareml/deeplearning/` is aligned to *PyTorch*, as seen in below 2 examples.

Example1: 
```
from bareml.machinelearning.utils.model_selection import train_test_split
from bareml.machinelearning.supervised import KernelRidge

# assume the data X, y are defined
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

reg = KernelRidge(alpha=1, kernel='rbf')
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg.score(X_test, y_test))
```

Example2:
```
from bareml.deeplearning import layers as nn
from bareml.deeplearning import functions as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=33856, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.flatten()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
```

## Installation 
```
$ pip install bareml
```
or
```
$ git clone https://github.com/shotahorii/bareml.git
$ cd bareml
$ python setup.py install
```

## Dependencies 

**Mandatory**
- numpy  

**Optional**
- cupy
- PIL
- matplotlib

## Examples
#### Generating handwriting digits by GAN
[[Google Colab]](https://github.com/shotahorii/bareml/blob/master/examples/GAN.ipynb)

![gif](https://media.giphy.com/media/FaQuqE6Otws0EL8RQ5/giphy.gif)

#### Cart Pole Problem with Q-Learning
[[Notebook]](https://github.com/shotahorii/bareml/blob/master/examples/q_learning.ipynb)

![gif](https://media.giphy.com/media/0YSkWnyRmFdLth4YMg/giphy.gif)

#### Clustering by DBSCAN
[[Notebook]](https://github.com/shotahorii/bareml/blob/master/examples/DBSCAN.ipynb)

![gif](https://media.giphy.com/media/KUAzkpSBJ4QmPsX9Kp/giphy.gif)

## Implementations 

### Deep Learning
- [Pytorch-like Deep Learning Framework](https://github.com/shotahorii/bareml/blob/master/bareml/deeplearning/)

### Supervised Learning
- [Bernoulli Naive Bayes](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/naive_bayes.py)
- [Decision Trees](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/decision_trees.py)
- [Elastic Net](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/linear_regression.py)
- [Gaussian Naive Bayes](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/naive_bayes.py)
- [Generalised Linear Model](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/glm.py)
- [K Nearest Neighbors](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/knn.py)
- [Kernel Ridge Regression](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/kernel_regression.py)
- [Lasso Regression](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/linear_regression.py)
- [Linear Regression](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/linear_regression.py)
- [Logistic Regression](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/logistic_regression.py)
- [Perceptron](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/perceptron.py)
- [Poisson Regression](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/glm.py)
- [Ridge Regression](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/supervised/linear_regression.py)

### Unsupervised Learning
- [KMeans (KMeans++)](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/unsupervised/kmeans.py)
- [DBSCAN](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/unsupervised/dbscan.py)

### Ensemble Learning 
- [AdaBoost](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/adaboost.py)
- [AdaBoost M1](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/adaboost.py)
- [AdaBoost Samme](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/adaboost.py)
- [AdaBoost RT](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/adaboost.py)
- [AdaBoost R2](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/adaboost.py)
- [Bagging](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/baggings.py)
- [Gradient Boosting](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/gradient_boosting.py)
- [Random Forest](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/baggings.py)
- [Stacking](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/stacking.py)
- Voting
- [XGBoost](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/ensemble/xgboost.py)

### Utilities
- [Preprocessing (Scaler, Encoder etc)](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/utils/preprocessing.py)
- [Metrics](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/utils/metrics.py)
- [Kernel functions](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/utils/kernels.py)
- [Probability Distributions](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/utils/probabilities.py)
- [Model Selection (KFold etc)](https://github.com/shotahorii/bareml/blob/master/bareml/machinelearning/utils/model_selection.py)


## References 
- Deep learning programs are based on O'Reilly Japan's book "Deep learning from scratch 3" (Koki Saitoh) and its implementation [Dezero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).
- References of machine learning programs are documented in each source file, but mostly based on original papers, "Pattern Recognition and Machine Learning" (Christopher M. Bishop) and/or "Machine Learning: A Probabilistic Perspective" (Kevin P. Murphy).