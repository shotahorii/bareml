import numpy as np

import sys
sys.path.append('../')

from bareml.deeplearning import Tensor
from bareml.deeplearning.utils import numerical_diff


# -------------------------------------------------------------
# Benchmark functions
# -------------------------------------------------------------


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


# -------------------------------------------------------------
# Test cases
# -------------------------------------------------------------


def test_sphere():
    x = Tensor(np.array([1.0]))
    y = Tensor(np.array([1.0]))
    z = sphere(x, y)
    z.backward()

    num_grad_x = numerical_diff(lambda v: sphere(v, y), x)
    num_grad_y = numerical_diff(lambda v: sphere(x, v), y)

    assert np.allclose(x.grad.data, num_grad_x) and\
           np.allclose(y.grad.data, num_grad_y)


def test_matyas():
    x = Tensor(np.array([1.0]))
    y = Tensor(np.array([1.0]))
    z = matyas(x, y)
    z.backward()

    num_grad_x = numerical_diff(lambda v: matyas(v, y), x)
    num_grad_y = numerical_diff(lambda v: matyas(x, v), y)

    assert np.allclose(x.grad.data, num_grad_x) and\
           np.allclose(y.grad.data, num_grad_y)


def test_goldstein():
    x = Tensor(np.array([1.0]))
    y = Tensor(np.array([1.0]))
    z = goldstein(x, y)
    z.backward()

    num_grad_x = numerical_diff(lambda v: goldstein(v, y), x)
    num_grad_y = numerical_diff(lambda v: goldstein(x, v), y)

    assert np.allclose(x.grad.data, num_grad_x) and\
           np.allclose(y.grad.data, num_grad_y)
