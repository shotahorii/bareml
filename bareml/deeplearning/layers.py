import os
import weakref
from abc import ABCMeta, abstractmethod
import numpy as np
from .core import Parameter, Tensor, get_array_module, flatten, reshape, cupy, as_tensor
import bareml.deeplearning.functions as F
from .utils import pair
from .config import Config


# -------------------------------------------------------------
# Base class for all layers
# -------------------------------------------------------------


class Layer(metaclass=ABCMeta):
    """
    Base layer class. 
    Defines common behaviours across all layers.
    
    Forward path is like below.     
    --
    l = SomeLayer()
    y = l(x)
    --
    x can be either xp.ndarray or bareml.Tensor
    y is always bareml.Tensor

    Parameters in layers should be initialised with np (not xp) by default. 
    """
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
        """
        inputs: either xp.ndarray or bareml.Tensor
        outputs: bareml.Tensor
        """
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
        #xp = cupy if is_available() else np
        # init with numpy as default. to use cp, use to_gpu()
        self.b = None if not bias else Parameter(np.zeros(out_features, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_features, self.out_features
        # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        W_data = xp.random.randn(I, O).astype(self.dtype) * xp.sqrt(1 / I)
        self.W.data = W_data

    def set_W(self, W_data):
        if self.W.data.shape != W_data.shape:
            raise ValueError('W shape unmatch.')
        self.W.data = W_data

    def set_b(self, b_data):
        if self.b.data.shape != b_data.shape:
            raise ValueError('b shape unmatch.')
        self.b.data = b_data

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


# -------------------------------------------------------------
# Embedding
# -------------------------------------------------------------


class Embedding(Layer):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(None, name='weight')
        self._init_weight()

    def _init_weight(self):
        # init with numpy by default. to use cp, use to_gpu()
        self.weight.data = np.random.randn(self.num_embeddings, self.embedding_dim).astype(np.float32)

    def forward(self, idx):
        """
        Parameters
        ----------
        idx: np.ndarray (n, c)
            n: batch size
            c: number of contexts to be embedded 

        Returns
        -------
        y: bareml.Tensor (n, c, embedding_dim)
            n: batch size
            c: number of contexts to be embedded
            embedding_dim: self.embedding_dim
        """
        y = F.embedding(self.weight, idx)
        return y


# -------------------------------------------------------------
# RNN: RNNCell / RNN
# -------------------------------------------------------------


class RNNCell(Layer):
    """ An Elman RNN cell with tanh or ReLU non-linearity. """
    def __init__(self, hidden_size, input_size=None, bias=True, nonlinearity='tanh'):
        super().__init__()
        if nonlinearity == 'tanh':
            self.f = F.tanh
        elif nonlinearity == 'relu':
            self.f = F.relu
        else:
            raise ValueError('nonlinearity must be either "tanh" or "relu".')

        self.x2h = Linear(out_features=hidden_size, in_features=input_size, bias=bias)
        self.h2h = Linear(out_features=hidden_size, in_features=hidden_size, bias=None)

        # init weight & bias
        # xp = cupy if is_available() else np
        # init with numpy by default. to use cp, use to_gpu()
        W_x2h = np.random.randn(*self.x2h.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.x2h.set_W(W_x2h)
        W_h2h = np.random.randn(*self.h2h.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.h2h.set_W(W_h2h)
        if bias:
            b_x2h = np.random.randn(*self.x2h.b.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
            self.x2h.set_b(b_x2h)
            #b_h2h = np.random.randn(*self.h2h.b.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
            #self.h2h.set_b(b_h2h)

    def forward(self, x, h): 
        """
        Parameters
        ----------
        x: bareml.Tensor (n, in_size)
            n: batch size
            in_size: input size

        h: bareml.Tensor (n, hidden_size)
            n: batch size
            hidden_size: hidden size

        Returns
        -------
        h_new: bareml.Tensor (n, hidden_size)
            n: batch size
            hidden_size: hidden size
        """
        h_new = self.f(self.x2h(x) + self.h2h(h))
        return h_new


class RNNCellWithDropout(Layer):
    def __init__(self, hidden_size, input_size=None, bias=True, nonlinearity='tanh', dropout=0.5):
        super().__init__()
        self.rnn = RNNCell(hidden_size, input_size, bias, nonlinearity)
        self.dropout = Dropout(dropout)
    
    def forward(self, x, h):
        y = self.rnn(x, h)
        y = self.dropout(y)
        return y


class RNN(Layer):

    def __init__(self, hidden_size, input_size=None, num_layers=1, 
                 bias=True, nonlinearity='tanh', batch_first=False, dropout=0):
        super().__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.rnns = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            if dropout > 0 and i < num_layers - 1:
                rnn = RNNCellWithDropout(hidden_size, in_size, bias, nonlinearity, dropout)
            else:
                rnn = RNNCell(hidden_size, in_size, bias, nonlinearity)
            setattr(self, 'rnn'+str(i), rnn)
            self.rnns.append(rnn)

    def forward(self, xs, h_0):
        """
        Parameters
        ----------
        xs: bareml.Tensor (l, batch_size, in_size)
            l: length of sequence
            batch_size: batch size
            in_size: input size
            if self.batch_first, shape is (batch_size, l, in_size)

        h_0: bareml.Tensor (num_layers, batch_size, hidden_size)
            num_layers: num_layers
            batch_size: batch size
            hidden_size: hidden_size

        Returns
        -------
        hs: bareml.Tensor (l, batch_size, hidden_size)
            l: length of sequence
            batch_size: batch size
            hidden_size: hidden_size
            if self.batch_first, shape is (batch_size, l, hidden_size)

        h_n: bareml.Tensor (num_layers, batch_size, hidden_size)
            num_layers: num_layers
            batch_size: batch size
            hidden_size: hidden_size
        """

        if self.batch_first:
            xs = xs.transpose(1,0,2)

        len_seq = xs.shape[0]
        batch_size = xs.shape[1]
        
        xp = get_array_module(xs)
        h_n = Tensor(xp.zeros((len(self.rnns), batch_size, self.hidden_size),dtype=np.float32))
        for k, rnn in enumerate(self.rnns):
            h_k = h_0[k]
            hs = Tensor(xp.zeros((len_seq, batch_size, self.hidden_size),dtype=np.float32))
            for i, x in enumerate(xs):
                h_k = rnn(x, h_k)
                hs[i] = h_k
            xs = hs
            h_n[k] = h_k
        
        if self.batch_first:
            hs = hs.transpose(1,0,2)
        return hs, h_n


class LSTMCell(Layer):
    """ A LSTM cell with tanh or ReLU non-linearity. """
    def __init__(self, hidden_size, input_size=None, bias=True):
        super().__init__()

        self.x2f = Linear(out_features=hidden_size, in_features=input_size, bias=bias)
        self.x2i = Linear(out_features=hidden_size, in_features=input_size, bias=bias)
        self.x2o = Linear(out_features=hidden_size, in_features=input_size, bias=bias)
        self.x2g = Linear(out_features=hidden_size, in_features=input_size, bias=bias)
        self.h2f = Linear(out_features=hidden_size, in_features=hidden_size, bias=False)
        self.h2i = Linear(out_features=hidden_size, in_features=hidden_size, bias=False)
        self.h2o = Linear(out_features=hidden_size, in_features=hidden_size, bias=False)
        self.h2g = Linear(out_features=hidden_size, in_features=hidden_size, bias=False)

        # init weight & bias
        # xp = cupy if is_available() else np
        # init with numpy by default. to use cp, use to_gpu()
        W_x2f = np.random.randn(*self.x2f.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.x2f.set_W(W_x2f)
        W_x2i = np.random.randn(*self.x2i.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.x2i.set_W(W_x2i)
        W_x2o = np.random.randn(*self.x2o.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.x2o.set_W(W_x2o)
        W_x2g = np.random.randn(*self.x2g.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.x2g.set_W(W_x2g)
        W_h2f = np.random.randn(*self.h2f.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.h2f.set_W(W_h2f)
        W_h2i = np.random.randn(*self.h2i.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.h2i.set_W(W_h2i)
        W_h2o = np.random.randn(*self.h2o.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.h2o.set_W(W_h2o)
        W_h2g = np.random.randn(*self.h2g.W.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.h2g.set_W(W_h2g)

        if bias:
            b_x2f = np.random.randn(*self.x2f.b.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
            self.x2f.set_b(b_x2f)
            b_x2i = np.random.randn(*self.x2i.b.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
            self.x2i.set_b(b_x2i)
            b_x2o = np.random.randn(*self.x2o.b.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
            self.x2o.set_b(b_x2o)
            b_x2g = np.random.randn(*self.x2g.b.shape).astype(np.float32) * np.sqrt(1 / hidden_size)
            self.x2g.set_b(b_x2g)

    def forward(self, x, h, c): 
        """
        Parameters
        ----------
        x: bareml.Tensor (n, in_size)
            n: batch size
            in_size: input size

        h: bareml.Tensor (n, hidden_size)
            n: batch size
            hidden_size: hidden size

        Returns
        -------
        h_new: bareml.Tensor (n, hidden_size)
            n: batch size
            hidden_size: hidden size
        """
        f = F.sigmoid(self.x2f(x) + self.h2f(h))
        i = F.sigmoid(self.x2i(x) + self.h2i(h))
        o = F.sigmoid(self.x2o(x) + self.h2o(h))
        g = F.tanh(self.x2g(x) + self.h2g(h))

        c_new = (f * c) + (i * g)
        h_new = o * F.tanh(c_new)
        return h_new, c_new


class LSTMCellWithDropout(Layer):
    def __init__(self, hidden_size, input_size=None, bias=True, dropout=0.5):
        super().__init__()
        self.lstm = LSTMCell(hidden_size, input_size, bias)
        self.dropout = Dropout(dropout)
    
    def forward(self, x, h, c):
        new_h, new_c = self.lstm(x, h, c)
        new_h = self.dropout(new_h)
        return new_h, new_c


class LSTM(Layer):

    def __init__(self, hidden_size, input_size=None, num_layers=1, 
                 bias=True, batch_first=False, dropout=0):
        super().__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.lstms = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            if dropout > 0 and i < num_layers - 1:
                lstm = LSTMCellWithDropout(hidden_size, in_size, bias, dropout)
            else:
                lstm = LSTMCell(hidden_size, in_size, bias)
            setattr(self, 'lstm'+str(i), lstm)
            self.lstms.append(lstm)

    def forward(self, xs, h_0, c_0):
        """
        Parameters
        ----------
        xs: bareml.Tensor (l, batch_size, in_size)
            l: length of sequence
            batch_size: batch size
            in_size: input size
            if self.batch_first, shape is (batch_size, l, in_size)

        h_0: bareml.Tensor (num_layers, batch_size, hidden_size)
            num_layers: num_layers
            batch_size: batch size
            hidden_size: hidden_size

        Returns
        -------
        hs: bareml.Tensor (l, batch_size, hidden_size)
            l: length of sequence
            batch_size: batch size
            hidden_size: hidden_size
            if self.batch_first, shape is (batch_size, l, hidden_size)

        h_n: bareml.Tensor (num_layers, batch_size, hidden_size)
            num_layers: num_layers
            batch_size: batch size
            hidden_size: hidden_size
        """

        if self.batch_first:
            xs = xs.transpose(1,0,2)

        len_seq = xs.shape[0]
        batch_size = xs.shape[1]
        
        xp = get_array_module(xs)
        h_n = Tensor(xp.zeros((len(self.lstms), batch_size, self.hidden_size),dtype=np.float32))
        c_n = Tensor(xp.zeros((len(self.lstms), batch_size, self.hidden_size),dtype=np.float32))
        for k, lstm in enumerate(self.lstms):
            h_k = h_0[k]
            c_k = c_0[k]
            hs = Tensor(xp.zeros((len_seq, batch_size, self.hidden_size),dtype=np.float32))
            for i, x in enumerate(xs):
                h_k, c_k = lstm(x, h_k, c_k)
                hs[i] = h_k
            xs = hs
            h_n[k] = h_k
            c_n[k] = c_k
        
        if self.batch_first:
            hs = hs.transpose(1,0,2)
        return hs, h_n, c_n