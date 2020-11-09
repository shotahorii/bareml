from .core import cupy

def is_available():
    return cupy is not None