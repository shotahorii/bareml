import os
import weakref
from abc import ABCMeta, abstractmethod
import numpy as np
from .core import Parameter, get_array_module, flatten, reshape, cupy
import baredl.functions as F
from .utils import pair
from .config import Config
from .cuda import is_available


# -------------------------------------------------------------
# Base class for all layers
# -------------------------------------------------------------


class Layer(metaclass=ABCMeta):
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    @abstractmethod
    def forward(self, inputs):
        pass

    def parameters(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.parameters()
            else:
                yield obj

    def children(self):
        layers = []
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                layers.append(obj)
                sublayers = obj.children()
                for sublayer in sublayers:
                    layers.append(sublayer)
        
        return iter(layers)


    def train(self, mode=True):
        Config.training = mode

    def eval(self):
        self.train(mode=False)

    def zero_grad(self):
        for param in self.parameters():
            param.cleargrad()

    def to_cpu(self):
        for param in self.parameters():
            param.to_cpu()

    def to_gpu(self):
        for param in self.parameters():
            param.to_gpu()

    def to(self, device):
        if device=='cpu':
            self.to_cpu()
        elif device=='cuda':
            self.to_gpu()
        else:
            raise ValueError('device can be either "cpu" or "cuda".')  

        return self

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu() # always save as np.ndarray not cp.ndarray

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            # if the save operation is interrupted by user input (such as Ctrl+C)
            # then remove the work-in-progress file
            if os.path.exists(path):
                os.remove(path) 
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# -------------------------------------------------------------
# Base class for models
# -------------------------------------------------------------


class Module(Layer):
    pass


# -------------------------------------------------------------
# Sequential
# -------------------------------------------------------------


class Sequential(Layer):
    def __init__(self, *layers):
        if not layers:
            raise ValueError('At least one layer needed.')
        elif not all([isinstance(l, Layer) for l in layers]):
            raise ValueError('Every input needs to be a Layer instance.')

        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
            
    def forward(self, x):  
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------------------------------------------
# Linear / Dropout / Flatten
# -------------------------------------------------------------


class Linear(Layer):
    def __init__(self, out_features, in_features=None, bias=True, dtype=np.float32):
        super().__init__()
        self.in_features = in_features # we can leave this None, and get from data in forward
        self.out_features = out_features
        self.dtype = dtype

        # init W. if in_features not specified, init later (when forward called)
        self.W = Parameter(None, name='W')
        if self.in_features is not None:
            self._init_W()
        # init bias
        xp = cupy if is_available() else np
        self.b = None if not bias else Parameter(xp.zeros(out_features, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_features, self.out_features
        # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        W_data = xp.random.randn(I, O).astype(self.dtype) * xp.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_features = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)
        
        y = F.linear(x, self.W, self.b)
        return y


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        y = F.dropout(x, dropout_ratio=self.p)
        return y


class Dropout2d(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        y = F.dropout2d(x, dropout_ratio=self.p)
        return y


class Flatten(Layer):
    def forward(self, x):
        y = flatten(x)
        return y


class Reshape(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        y = reshape(x, self.shape)
        return y


# -------------------------------------------------------------
# Activation
# -------------------------------------------------------------


class ReLU(Layer):
    def forward(self, x):
        y = F.relu(x)
        return x


class LeakyReLU(Layer):
    def __init__(self, slope=0.2):
        super().__init__()
        self.slope = slope

    def forward(self, x):
        y = F.leaky_relu(x, self.slope)
        return y


class Sigmoid(Layer):
    def forward(self, x):
        y = F.sigmoid(x)
        return y


class Tanh(Layer):
    def forward(self, x):
        y = F.tanh(x)
        return y


# -------------------------------------------------------------
# Conv2d / ConvTranspose2d
# -------------------------------------------------------------


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=True, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
        
        if bias:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
        else:
            self.b = None

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class ConvTranspose2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=True, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
        
        if bias:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
        else:
            self.b = None

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        y = F.conv_transpose2d(x, self.W, self.b, self.stride, self.pad)
        return y


# -------------------------------------------------------------
# Max Pooling
# -------------------------------------------------------------


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding

    def forward(self, x):
        y = F.max_pool2d(x, self.kernel_size, self.stride, self.pad)
        return y


# -------------------------------------------------------------
# Batch Normalisation
# -------------------------------------------------------------


class BatchNorm2d(Layer):
    def __init__(self, num_features=None, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps=eps
        self.momentum = momentum
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = get_array_module(x)

        D = x.shape[1] if self.num_features is None else self.num_features

        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def forward(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)

        y = F.batch_norm(x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data,decay=1.0-self.momentum, eps=self.eps)
        return y


class BatchNorm1d(BatchNorm2d):
    """ Above implementation works for 1d i.e. x is (N,C) shape """
    pass


# -------------------------------------------------------------
# Upsample
# -------------------------------------------------------------


class Upsample(Layer):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        y = F.upsample(x, self.scale_factor, self.mode)
        return y