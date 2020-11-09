import math
import numpy as np

import sys
sys.path.append('../')

from bareml.deeplearning import Tensor
from bareml.deeplearning.functions import exp
from bareml.deeplearning.utils import numerical_diff


def test_exp():
    var = np.random.rand(1) # var is a np.array([x])
    print('testtt')
    print(var)

    x = Tensor(np.array(var))
    y = exp(x)
    expected = math.exp(var)
    assert np.allclose(y.data, expected)


def test_exp_trad():
    var = np.random.rand(1) # var is a np.array([x])

    x = Tensor(np.array(var))
    y = exp(x)
    y.backward()
    num_grad = numerical_diff(exp, x)
    assert np.allclose(x.grad.data, num_grad)
