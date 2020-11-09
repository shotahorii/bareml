import numpy as np
from .core import Tensor, Function, reverse_broadcast_to, get_array_module, sum, clip, repeat_interleave
from .utils import logsumexp, pair, im2col_array, col2im_array, get_deconv_outsize
from .config import Config


# -------------------------------------------------------------
# Basic functions: exp / log / sin / cos / tanh
# -------------------------------------------------------------


class Exp(Function):
    def forward(self, x):
        """
        Parameters
        ----------
        x: xp.ndarray (baredl.Tensor.data)

        Returns
        -------
        y: xp.ndarray
        """
        xp = get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor (baredl.Tensor.grad)

        Returns
        -------
        gx: baredl.Tensor
        """
        #x = self.inputs[0].data
        #gx = np.exp(x) * gy
        y = self.outputs[0]() # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Sin(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # weakref
        gx = gy * (1 - y ** 2)
        return gx


def tanh(x):
    return Tanh()(x)


# -------------------------------------------------------------
# Activation functions: sigmoid / softmax
# -------------------------------------------------------------


class Sigmoid(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = 1 / (1 + xp.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1-y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # weakref
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis=axis

    def forward(self, x):
        """
        https://blog.feedly.com/tricks-of-the-trade-logsumexp/
        """
        log_z = logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]() #weakref
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class ReLU(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs 
        mask = x.data > 0 
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        x, = self.inputs 
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


# -------------------------------------------------------------
# Loss functions: MSELoss, CrossEntropyLoss
# -------------------------------------------------------------


class MSELoss(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2) / len(diff)
        return y 
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mse_loss(x0, x1):
    return MSELoss()(x0, x1)


class CrossEntropyLoss(Function):
    """
    Cross Entropy Loss of Softmax.
    """
    def forward(self, x, t):
        """
        Parameters
        ----------
        x: np.ndarray (n, c)
            n: number of samples
            c: number of classes

        t: np.ndarray (n,) or (n,1)
            n: number of samples
            label index of true class of each sample
            
        Returns
        -------
        y: np.scalar
        """
        N = x.shape[0]
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z # log softmax
        # for each sample, get only prob of the true label.
        # e.g. if log_p = [[-1.4, -2.4, -0.4], [-3.2, -2.2, -0.2]]
        #      and t = [0,1]
        #      then log_p[np.arange(N), t.ravel()] = [-1.4, -2.2]
        log_p = log_p[np.arange(N), t.ravel()]
        # get average
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs 
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        xp = get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def cross_entropy(x, t):
    return CrossEntropyLoss()(x, t)


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy_with_logits(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = sigmoid(p)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# -------------------------------------------------------------
# Other functions: linear, dropout
# -------------------------------------------------------------


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else reverse_broadcast_to(gy, b.shape)
        gx = gy @ W.T # gx = matmul(gy, W.T)
        gW = x.T @ gy # gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Dropout(Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        if Config.training:
            xp = get_array_module(x)
            self.mask = xp.random.rand(*x.shape) > self.dropout_ratio
            self.scale = xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            y = x * self.mask / self.scale
            return y
        else:
            self.mask, self.scale = 1.0, 1.0 # practically no need to do this though.
            return x

    def backward(self, gy):
        gx = gy * self.mask / self.scale
        return gx


def dropout(x, dropout_ratio=0.5):
    return Dropout(dropout_ratio)(x)


class Dropout2d(Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        if Config.training:
            xp = get_array_module(x)

            channels_removed = xp.random.rand(x.shape[1]) <= self.dropout_ratio
            self.mask = xp.ones(x.shape)
            self.mask[:,channels_removed] = 0.0
            self.scale = xp.array(1.0 - channels_removed.sum()/len(channels_removed)).astype(x.dtype)

            y = x * self.mask / self.scale
            return y
        else:
            self.mask, self.scale = 1.0, 1.0 # practically no need to do this though.
            return x

    def backward(self, gy):
        gx = gy * self.mask / self.scale
        return gx


def dropout2d(x, dropout_ratio=0.5):
    return Dropout2d(dropout_ratio)(x)


# -------------------------------------------------------------
# Conv functions: 
# -------------------------------------------------------------


class Conv2d(Function):

    def __init__(self, stride=1, padding=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(padding)

    def forward(self, x, W, b):
        """
        Parameters
        ----------
        x: xp.ndarray (N, C, H, Width)
            N: number of samples (images)
            C: number of channels
            H: height of the images
            Width: width of the images
            e.g. x with shape (3,2,3,4) is like below.
                np.array([
                    # sample1
                    [
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]], # channel1 (3*4 matrix)
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]]  # channel2 (3*4 matrix)
                    ],
                    # sample2
                    [
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]], # channel1 (3*4 matrix)
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]]  # channel2 (3*4 matrix)
                    ],
                    # sample 3
                    [
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]], # channel1 (3*4 matrix)
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]]  # channel2 (3*4 matrix)
                    ],
                ])

        W: xp.ndarray (OC, C, KH, KW)
            OC: number of output channels
            C: number of channels 
            KH: height of the kernel (filter)
            KW: width of the kernel (filter)
            e.g. W with shape (4,2,2,2) is like below.
                np.array([
                    # output channel 1
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]]  # channel2 (2*2 kernel)
                    ],
                    # output channel 2
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]]  # channel2 (2*2 kernel)
                    ],
                    # output channel 3
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]]  # channel2 (2*2 kernel)
                    ],
                    # output channel 4
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]  # channel2 (2*2 kernel)
                    ],
                ])

        b: xp.ndarray (OC,)
            OC: number of output channels
            e.g. b with shape (4,) is like below.
                np.array([0, 0, 0, 0]) # each 0 is for each output channel

        Returns
        -------
        y: xp.ndarray (N, OC, OH, OW)
            N: number of samples (images)
            OC: number of output channels
            OH: height of output images
            OW: width of output images
            e.g. with all the examples above, y's shape would be (3,4,2,3) like below.
                np.array([
                    # sample1
                    [
                        [[0,0,0],[0,0,0]], # output channel1 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel2 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel3 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel4 (2*3 matrix)
                    ],
                    # sample2
                    [
                        [[0,0,0],[0,0,0]], # output channel1 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel2 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel3 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel4 (2*3 matrix)
                    ],
                    # sample 3
                    [
                        [[0,0,0],[0,0,0]], # output channel1 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel2 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel3 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel4 (2*3 matrix)
                    ],
                ])
        """
        xp = get_array_module(x)

        KH, KW = W.shape[2:]

        # col is a xp.ndarray (N, C, KH, KW, OH, OW)
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        # col is a xp.ndarray (N,  C, KH, KW, OH, OW)
        # W   is a xp.ndarray (OC, C, KH, KW)
        # y   is a xp.ndarray (N, OH, OW, OC)
        axes = ((1, 2, 3), (1, 2, 3))
        y = xp.tensordot(col, W, axes=axes)

        if b is not None:
            # y is a xp.ndarray (N, OH, OW, OC)
            # b is a xp.ndarray (OC,)
            # here, b is broadcasted to (1, 1, 1, OC) then (N, OH, OW, OC)
            y += b

        # (N, OH, OW, OC) to (N, OC, OH, OW)
        y = xp.rollaxis(y, 3, 1)

        return y

    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor (N, OC, OH, OW)
            forward's output's grad.

        Returns 
        -------
        gx: baredl.Tensor (N, C, H, Width)

        gW: baredl.Tensor (OC, C, KH, KW)
        
        gb: baredl.Tensor (OC,)
        """
        x, W, b = self.inputs

        H, Width = x.shape[2], x.shape[3]
        
        gx = conv_transpose2d(gy, W, b=None, stride=self.stride, padding=self.pad, outsize=(H, Width))
        gW = Conv2dGradW(self)(x, gy)
        gb = None

        # note b (input) is stored as a Tensor even if it's None.
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, padding=0):
    return Conv2d(stride, padding)(x, W, b)


class ConvTranspose2d(Function):
    
    def __init__(self, stride=1, padding=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(padding)
        self.outsize = outsize

    def forward(self, x, W, b):
        """
        Parameters
        ----------
        x: xp.ndarray (N, C, H, Width)
            N: number of samples (images)
            C: number of channels
            H: height of the images
            Width: width of the images
        
        W: xp.ndarray (C, OC, KH, KW)
            C: number of channels
            OC: number of output channels
            KH: height of the kernel (filter)
            KW: width of the kernel (filter)
            Note that shape of the input W in Conv2d is (OC, C, KH, KW)
            whereas this input W is (C, OC, KH, KW). This is because Conv2d's 
            input channel C is the output channel OC from the perspective of ConvTranspose2d.
            And Conv2d's output chanel OC is the input chanel C of the ConvTranspose2d.
            So, same W, but the way we call C, OC is opposite from the perspective of Conv2d 
            vs ConvTranspose2d.

        b: xp.ndarray (OC,)
            OC: number of output channels
            e.g. b with shape (4,) is like below.
                np.array([0, 0, 0, 0]) # each 0 is for each output channel

        Returns
        -------
        y: xp.ndarray (N, OC, OH, OW)
        """
        xp = get_array_module(x)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape 

        if self.outsize is None:
            OH = get_deconv_outsize(H, KH, SH, PH)
            OW = get_deconv_outsize(W, KW, SW, PW)
        else:
            OH, OW = pair(self.outsize)

        # Note that ConvTranspose2d is reverse operation of Conv2d.
        # So, below (N, OC, OH, OW) is Conv2d's (N, C, H, W)
        img_shape = (N, OC, OH, OW)

        # Weight's shape is (C, OC, KH, KW)
        # x's shape is      (N, C,  H,  W)
        # gcol's shape is   (OC, KH, KW, N, H, W)
        gcol = xp.tensordot(Weight, x, (0,1))

        # (OC, KH, KW, N, H, W) to (N, OC, KH, KW, H, W)
        gcol = xp.rollaxis(gcol, 3)

        # y's shape is (N, OC, OH, OW)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)

        if b is not None:
            # b's shape is (OC,)
            # since (OC,) cannot be broadcasted to (N, OC, OH, OW),
            # need to respahe to (1, OC, 1, 1) first. Then 
            # (1, OC, 1, 1) will be broadcasted to (N, OC, OH, OW)
            y += b.reshape((1, b.size, 1, 1))

        return y

    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor (N, OC, OH, OW)
            forward's output's grad.

        Returns 
        -------
        gx: baredl.Tensor (N, C, H, Width)

        gW: baredl.Tensor (C, OC, KH, KW)
        
        gb: baredl.Tensor (OC,)
        """
        x, W, b = self.inputs

        gx = conv2d(gy, W, b=None, stride=self.stride, padding=self.pad)
        gW = Conv2dGradW(self)(gy, x) # not (x, gy, self) but (gy, x, self)
        gb = None

        # note b (input) is stored as a Tensor even if it's None.
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def conv_transpose2d(x, W, b=None, stride=1, padding=0, outsize=None):
    return ConvTranspose2d(stride, padding, outsize)(x, W, b)

        
class Conv2dGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        KH, KW = W.shape[2:]
        self.kernel_size = (KH, KW)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        gW = xp.tensordot(gy, col, ((0,2,3),(0,4,5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        XH, XW = x.shape[2:]
        gx = conv_transpose2d(gy, gW, stride=self.stride, padding=self.pad,outsize=(XH, XW))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# -------------------------------------------------------------
# Pooling functions: 
# -------------------------------------------------------------


class MaxPool2d(Function):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        y = col.max(axis=2)
        self.indices = col.argmax(axis=2)

        return y

    def backward(self, gy):
        return MaxPool2dGrad(self)(gy)


def max_pool2d(x, kernel_size, stride=1, padding=0):
    return MaxPool2d(kernel_size, stride, padding)(x)


class MaxPool2dGrad(Function):
    def __init__(self, maxpool2d):
        self.maxpool2d = maxpool2d
        self.kernel_size = maxpool2d.kernel_size
        self.stride = maxpool2d.stride
        self.pad = maxpool2d.pad
        self.input_shape = maxpool2d.inputs[0].shape
        self.dtype = maxpool2d.inputs[0].dtype
        self.indices = maxpool2d.indices

    def forward(self, gy):
        xp = get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indices = (self.indices.ravel() + xp.arange(0, self.indices.size * KH * KW, KH * KW))

        gcol[indices] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False)

        return gx

    def backward(self, ggx):
        f = MaxPool2dWithIndices(self.maxpool2d)
        return f(ggx)


class MaxPool2dWithIndices(Function):

    def __init__(self, maxpool2d):
        self.kernel_size = maxpool2d.kernel_size
        self.stride = maxpool2d.stride
        self.pad = maxpool2d.pad
        self.input_shape = maxpool2d.inputs[0].shape
        self.dtype = maxpool2d.inputs[0].dtype
        self.indices = maxpool2d.indices

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indices = self.indices.ravel()
        col = col[np.arange(len(indices)), indices]
        return col.reshape(N, C, OH, OW)


# -------------------------------------------------------------
# Batch normalisation
# -------------------------------------------------------------   


class BatchNorm(Function):
    """
    https://leimao.github.io/blog/Batch-Normalization/
    https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/
    """
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = get_array_module(x)

        if Config.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_norm(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


# -------------------------------------------------------------
# Upsample
# -------------------------------------------------------------  


def upsample(x, scale_factor, mode='nearest'):
    """
    x: baredl.Tensor (N, C, W) or (N, C, H, W)

    scale_factor: int or tuple of ints
        if x is (N, C, W), scale_factor needs to be int or (int,)
        if x is (N, C, H, W), scale_factor can be int, (int,) or (int, int) 
    """
    if x.ndim != 3 and x.ndim != 4:
        raise ValueError('input needs to be 3d or 4d tensor.')

    if mode != 'nearest':
        raise ValueError('Currently only supporting mode="nearest"')

    # standardise the input scale_factor format
    if isinstance(scale_factor, tuple) and len(scale_factor)==1:
        scale_factor = scale_factor[0]
    scale_dim2, scale_dim3 = pair(scale_factor) 

    y = x.repeat_interleave(repeats=scale_dim2, dim=2)
    if x.ndim == 4:
        y = y.repeat_interleave(repeats=scale_dim3, dim=3)

    return y
        

