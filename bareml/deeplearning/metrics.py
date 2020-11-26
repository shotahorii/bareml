import numpy as np
from .core import Tensor, as_tensor, as_array, get_array_module

def accuracy(y, t):
    """
    y: baredl.Tensor or np.ndarray (n, c)
        n: number of samples
        c: number of classes
        Assuming it contains probabilities for each class
        e.g. [[0.1,0.3,0.6], [0.1,0.8,0.1], ...]

    t: baredl.Tensor or np.array (n,)
        n: number of samples
        Assuming it contains the true class label as index
        e.g. [2,1,1,0,2,0,...]
    """
    y, t = as_tensor(y), as_tensor(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Tensor(as_array(acc))


def cos_similarity(x, y, eps=1e-8):
    if isinstance(x, Tensor):
        x = x.data
    if isinstance(y, Tensor):
        y = y.data
    
    xp = get_array_module(x)
    nx = x / (xp.sqrt(xp.sum(x**2))+eps)
    ny = y / (xp.sqrt(xp.sum(y**2))+eps)
    return xp.dot(nx, ny)