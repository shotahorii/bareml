import weakref
from abc import ABCMeta, abstractmethod
import numpy as np
from .config import Config, using_config

try:
    import cupy as cp
    cupy = cp
    array_types = (np.ndarray, cp.ndarray)
except ImportError:
    cupy = None
    array_types = (np.ndarray)


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------


def as_array(x, array_module=np):
    """
    Convert scalar x to xp.array datatype.
    e.g. 3 -> np.array(3)

    Parameters
    ----------
    x: xp.ndarray (any shape), xp.scalar or scalar
    array_module: {numpy, cupy}
    """
    if np.isscalar(x):
        return array_module.array(x)
    return x


def as_tensor(obj):
    """
    Convert np.array object to Tensor.
    e.g. np.array([1,2]) -> Tensor([1,2])

    Parameters
    ----------
    obj: xp.ndarray (any shape) of real (-inf, inf)
    """
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)


def as_numpy(x):
    if isinstance(x, Tensor):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x

    if cupy is None:
        raise Exception('CuPy not loaded.')
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Tensor):
        x = x.data

    if cupy is None:
        raise Exception('CuPy not loaded.')
    return cp.asarray(x)


def get_array_module(x):
    if isinstance(x, Tensor):
        x = x.data

    if cupy is None:
        return np
    xp = cp.get_array_module(x)
    return xp


# -------------------------------------------------------------
# Tensor / Parameter 
# -------------------------------------------------------------


class Tensor:
    """
    Data container class

    Parameters
    ----------
    data: xp.ndarray (any shape) of real (-inf, inf)
    name: string
    """

    # to prioritise __radd__ of this class over np.ndarray's __add__
    # when np.array([2.0]) + Tensor(np.array([1.0]))
    # also same for __rmul__ vs np.ndarray's __mul__
    __array_priority__ = 200  
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None # function which generated this Tensor instance
        # generation indicates "depth" of the position of this Tensor
        # in the calculation graph. This is important when we perform
        # backprop in a complex calculation graph.
        self.generation = 0

    def __len__(self):
        """ define len(Tensor) """
        return len(self.data)

    def __repr__(self):
        """ define print(Tensor) """
        if self.data is None:
            return 'tensor(None)'
        p = str(self.data).replace('\n', '\n' + ' '*7) # 7 is length of "tensor("
        return 'tensor(' + p + ')'

    def __getitem__(self, slices):
        """ 
        define Tensor[...] 
        e.g. Tensor[:,2] , Tensor[1,1], Tensor[[0,1,1]]
        """
        return get_item(self, slices)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)
    
    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, other):
        return pow(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return transpose(self)

    def astype(self, dtype):
        self.data.dtype = dtype
        return self

    def flatten(self):
        return flatten(self)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return max(self, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return min(self, axis, keepdims)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def repeat_interleave(self, repeats=2, dim=None):
        return repeat_interleave(self, repeats, dim)

    def set_creator(self, func):
        self.creator = func
        # generation of this Tensor instance will be 1 step deeper 
        # than the function created this instance. 
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        """
        calculate gradients of ancestor Tensor instances by backprop.

        Parameters
        ----------
        retain_grad: bool
            If True, keep grad values of every single Tensor instance
            in the calculation graph. 
            If False, only keep grad values of "end node" Tensor instances.
            This is for memory efficiency purpose. In most cases, False is fine. 
        
        create_graph: bool
            This indicates if we need to keep calculation graph for gradients. 
            if True, we keep calculation graph for grad i.e. grad.backward() is available.
            This needs to be True only if you need to do double backprop. 
        """
        
        # "self.grad is None" means this Tensor is the starting point
        # of the backprop. Because if this Tensor instance is in the 
        # middle of backprop, self.grad should be already defined (not None)
        # by the time this backward is called. 
        # Init value is always 1 because, e.g. forward flow is "x -> z -> L"
        # then backprop is dL/dx = dL/dL * dL/dz * dz/dx 
        # where the starting point dL/dL is always 1. 
        if self.grad is None:
            xp = get_array_module(self.data)
            self.grad = Tensor(xp.ones_like(self.data)) # grad is also a Tensor! This allows us double backprop.

        # funcs is a list to store Function instances of which 
        # backward need to be called.
        funcs = []
        # seen_set is a set to store Function instances which 
        # we ran backward once. 
        # this is to prevent same Function instance's backward
        # is called and calculated multiple times by mistake. 
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)

                # sort Function instances in funcs by generation.
                # so that always Function instances in "deeper"
                # position's called first.
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # since funcs is sorted by generation, this always gives the func with the largest generation.
            
            # gradients of all outputs of f. The order of elements is corresponding to the order of ys = f.forward(*xs).
            # Note: we access like "output()"" not just "output" because it's a weakref.
            gys = [output().grad for output in f.outputs]

            # if create_graph is False (which is most likely), 
            # do not keep calculation graph for grad calculation.
            with using_config('enable_backprop', create_graph):

                # calculate the gradients of f's inputs Tensor instances
                # using gradients of f's outputs.
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple): # make sure gxs is a tuple format
                    gxs = (gxs,)
                
                # set f's input's gradient
                for x, gx in zip(f.inputs, gxs): # Note: order of f.inputs & order of gxs = f.backward(*gys) is corresponding. so zip works. 
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # this is the case when f's input x is also an input of 
                        # another Function instance, and already grad given by its backward.
                        # in that case, we add gradient from this f on top of it. 
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
            
            if not retain_grad: # Note: even not retain_grad, grad of end nodes will be still retained
                for y in f.outputs:
                    y().grad = None

    def to_cpu(self):
        if self.data is not None:
            self.data = as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = as_cupy(self.data)

    def to(self, device):
        if device=='cpu':
            self.to_cpu()
        elif device=='cuda':
            self.to_gpu()
        else:
            raise ValueError('device can be either "cpu" or "cuda".') 

        return self


class Parameter(Tensor):
    pass


# -------------------------------------------------------------
# Base class of functions
# -------------------------------------------------------------


class Function(metaclass=ABCMeta):
    """
    Base class of all functions defined in baredl, which operate/manipulate Tensor instances.
    Functions can take np.ndarray as a input but it will be converted into Tensor. 
    """

    def __call__(self, *inputs):
        """
        Perform the operation (specified in self.forward) on the data of the given
        Tensor instances. Return the result as a (list of) Tensor instance(s).

        Parameters
        ----------
        inputs: a tuple of one or more of Tensor or np.ndarray (any shape) of real (-inf, inf)

        Returns
        -------
        Outputs: a list of Tensor, or a Tensor
        """

        # make sure every input is Tensor datatype
        inputs = [as_tensor(input) for input in inputs]

        # take data (np.ndarray) from each input (Tensor)
        xs = [input.data for input in inputs] # xs: list of np.ndarray

        # perform operation on the data (np.ndarray)
        ys = self.forward(*xs) # ys: np.ndarray or tuple of np.ndarray
        if not isinstance(ys, tuple): # if ys is a np.ndarray, convert to a tuple
            ys = (ys,)

        # each element of the tuple ys is most likely to be a np.ndarray
        # but in case of it's a scalar, apply as_array() and then make it as a Tensor.
        outputs = [Tensor(as_array(y)) for y in ys]

        # Keeping references to inputs / outputs are for backprop purpose. 
        # This is always needed at training, but no need at inference (prediction). 
        # So when we call this at inference, turn off this block to reduce memory usage, using Config.
        # Also, when this Function instance is called in backward (i.e. calculation of gradient), 
        # we most likely don't need to store these information unless we need to do double backprop.
        if Config.enable_backprop:
            # self.generation value indicates how deep this function is in the entire calc graph.
            # in the other words, how far this function is from the first input,
            # or how close this function is from the final outputs of the calc graph.
            # set this Function instance's generation as same as the biggest generation out of 
            # its inputs. 
            # this is because we want to make sure to run backward of Function instances in
            # deeper places before running backward of Function instances in less deep place 
            # in the calc graph.
            self.generation = np.max([input.generation for input in inputs])

            # set all outputs' creator as this Function instance.
            # so that those outputs can refer to this when backprop. 
            for output in outputs:
                output.set_creator(self)

            # store the reference to the inputs, so that 
            # this Function instance can refer to them when backprop.
            self.inputs = inputs

            # also needs to store the reference to the outputs,
            # as we use outputs' grads for self.backward().
            # however, as outputs have also reference to this Function instance.
            # to prevent a circular reference issue, we use weakref.
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    @abstractmethod
    def forward(self, x):
        """ x should be one or more of np.ndarray (input Tensor's data) """
        pass

    @abstractmethod
    def backward(self, gy):
        """ gy should be one or more of Tensor (output Tensor's grad) """
        pass


# -------------------------------------------------------------
# Basic arithmetic operations: add / mul / neg / sub / rsub / div / rdiv / pow
# -------------------------------------------------------------


class Add(Function):
    """ Class implementation for add function below. """

    def forward(self, x0, x1):
        """ See doc of add function below. """

        # if x0 and x1 have different shape, np automatically broadcast.
        # e.g. np.array([1,2,3]) + np.array(1) = np.array([2,3,4])
        y = x0 + x1 
        return y 
    
    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor
            A grad from former backprop calculation.
            e.g. Assume L = 2y, y = x0 + x1
                 Then, dL/dx = dL/dy * dy/dx
                 The parameter gy is equivalent to dL/dy in this example. 
                 As dy/dx0 and dy/dx1 are both 1, dL/dx0 == dL/dx1 == dL/dy
            So, gx0, gx1 = gy, gy as written below.
        """
        gx0, gx1 = gy, gy

        # if broadcast happened in forward i.e. x0.shape != x1.shape, 
        # we need to reverse broadcast gx0 or gx1 back to the original shape.
        x0, x1 = self.inputs
        if x0.shape != x1.shape:
            gx0 = reverse_broadcast_to(gx0, x0.shape)
            gx1 = reverse_broadcast_to(gx1, x1.shape)

        return gx0, gx1


def add(x0, x1):
    """
    Basic arithmetic '+' operation for bareml.Tensor.
    Used only to override Tensor class's __add__ and __radd__.

    Parameters
    ----------
    x0: xp.ndarray (any shape)
    x1: xp.ndarray (any shape) or scalar
        Note that since this function is only called 
        via Tensor class's __add__ and __radd__, 
        x0 is always xp.ndarray (Tensor.data) but 
        x1 can be scalar. e.g. Tensor(1) + 3
    """
    x1 = as_array(x1, get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        """ x0, x1: np.ndarray (any shape) """
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = reverse_broadcast_to(gx0, x0.shape)
            gx1 = reverse_broadcast_to(gx1, x1.shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        # for broadcast
        x0, x1 = self.inputs
        if x0.shape != x1.shape:
            gx0 = reverse_broadcast_to(gx0, x0.shape)
            gx1 = reverse_broadcast_to(gx1, x1.shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (- x0 / (x1 ** 2))
        if x0.shape != x1.shape:
            gx0 = reverse_broadcast_to(gx0, x0.shape)
            gx1 = reverse_broadcast_to(gx1, x1.shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = self.c * (x ** (self.c - 1)) * gy
        return gx


def mul(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    Since this function is only used to override Tensor class's 
    __mul__ and __rmul__, 
    input x0 is always a Tensor's data. Which means always np.ndarray.
    In contrast, x1 can be a scalar. e.g. Tensor(np.array(1)) * 3.0
    """
    x1 = as_array(x1, get_array_module(x0.data))
    return Mul()(x0, x1)


def neg(x):
    """ 
    x: np.ndarray (any shape)
    Since this function is only used to override Tensor class's 
    __neg__, 
    input x is always a Tensor's data. Which means always np.ndarray.
    """
    return Neg()(x)


def sub(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    Since this function is only used to override Tensor class's 
    __sub__, 
    input x0 is always a Tensor's data. Which means always np.ndarray.
    In contrast, x1 can be a scalar. e.g. Tensor(np.array(1)) - 3.0
    """
    x1 = as_array(x1, get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    Since this function is only used to override Tensor class's 
    __rsub__, 
    input x0 is always a Tensor's data. Which means always np.ndarray.
    In contrast, x1 can be a scalar. e.g. 3.0 - Tensor(np.array(1))
    """
    x1 = as_array(x1, get_array_module(x0.data))
    return Sub()(x1, x0)


def div(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    """
    x1 = as_array(x1, get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    """
    x1 = as_array(x1, get_array_module(x0.data))
    return Div()(x1, x0)


def pow(x,c):
    """ 
    x: np.ndarray (any shape) or scalar 
    """
    return Pow(c)(x)


# -------------------------------------------------------------
# Tensor manipulation: get_item / reshape / transpose
# -------------------------------------------------------------


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = get_array_module(gy)
        gx = xp.zeros_like(self.in_shape)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_tensor(x)
    return Reshape(shape)(x)


def flatten(x):
    return reshape(x, (x.shape[0], -1))


def expand_dims(x, axis):
    x = as_tensor(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes==None:
            return tanspose(gy)

        inv_axes = tuple(np.argsort(self.axes)) # should I use tuple(np.argsort([ax % len(self.axes) for ax in self.axes])) ??
        return transpose(x, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


# -------------------------------------------------------------
# Tensor operations: 
# -------------------------------------------------------------


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        
        # Here, we'd like to reshape gy back to original x shape
        # by using broadcast_to.
        # To do so, we firstly want to make sure input gy has  
        # the shape which is broadcast-able to original x shape.
        # What does this mean? For example, 
        # x = Tensor(np.array([[1,2,3],[4,5,6]]))
        # y = sum(x, axis=1, keepdims=False)
        # then, y is Tensor([6, 15])
        # hence, gy is Tensor([1, 1])
        # now, x.shape is (2,3)
        # Tensor([1, 1]) cannot be broadcasted to Tensor([[1,1,1],[1,1,1]])
        # i.e. gy needs to be Tensor([[1], [1]]) instead of Tensor([1, 1])        
        gy = self._reshape_broadcastable(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

    def _reshape_broadcastable(self, gy, x_shape, axis, keepdims):
        """
        Reshape gradient appropriately for bareml.sum's backward.
        
        Parameters
        ----------
        gy: baredl.Tensor
            Gradient tensor from the output by backprop.
        
        x_shape: tuple
            Shape used at sum function's forward.
            
        axis: None or int or tuple of int
            Axis used at sum function's forward.
        
        keepdims: bool
            Keepdims used at sum function's forward.
        
        Returns
        -------
        baredl.Tensor
            Gradient tensor which is reshaped appropriately
        """
        ndim = len(x_shape)

        # standardise axis
        if axis is None:
            tupled_axis = None
        elif not isinstance(axis, tuple): # only one axis e.g. axis=0
            tupled_axis = (axis,)
        else: # multiple axes e.g. axis=(0,1)
            tupled_axis = axis

        if ndim == 0 or tupled_axis is None or keepdims:
            # if ndim==0, x is a scalar, then y and gy are also scalar.
            # hence no problem with broadchasting.
            # if tupled_axis is None, y and gy are always scalar. 
            # hence no problem with broadcasting.
            # if keepdims==True, shape after sum is kept as same. 
            # so no need to reshape to broadcast. 
            return gy
        else:
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis] # just deal with negative axes
            shape = list(gy.shape)
            for ax in sorted(actual_axis): # fill the summed dimentions with 1. So that broadcastable.
                shape.insert(ax, 1)
            
            return gy.reshape(shape) 


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    """ 
    Class implementation for broadcast_to function. 
    See doc of broadcast_to function below. 
    """
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        """ See doc of broadcast_to function below. """
        self.x_shape = x.shape
        xp = get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        """
        Backward computation of broadcast_to is to sum up 
        elements in gy to its original x.
        Let's understand this with a couple of examples. 
        e.g.1 x is a scalar
              y = broadcast_to(x, (2,)) = [x, x]
              L = 2y[0] + 3y[1]
              Then, dL/dx = dL/dy0 * dy0/dx + dL/dy1 * dy1/dx
              Note that dy0/dx == dy1/dx == 1 because y[0] and y[1]
              are just exact copies of x. 
              Hence, dL/dx = dL/dy0 + dL/dy1
        e.g.2 x = [x0, x1, x2]
              y = broadcast_to(x, (2,3)) = [[x0, x1, x2],[x0, x1, x2]]
              L = 2y[0] + 3y[1]
              dL/dx0 = dL/dy0 * dy0/dx0 + dL/dy1 * dy1/dx0
              Note that dy0/dx0 == dy1/dx0 == [1,0,0]
              Hence, dL/dx0 = (dL/dy0 + dL/dy1) * [1,0,0]

        Parameters
        ----------
        gy: baredl.Tensor
            A grad from former backprop calculation.
        """
        gx = reverse_broadcast_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    """
    Work as same as np.broadcast_to.
    e.g. np.broadcast_to([5,6,7], (2,3)) -> [[5,6,7],[5,6,7]]
    
    Parameters
    ----------
    x: baredl.Tensor or xp.ndarray (any shape)
    
    shape: tuple of ints 
        Shape that x is broadcasted to. 
        -> Limitation of shape that x can be broadcasted to.
           In this example, assume shape is 2-dimention, say, shape = (r, c)
           in that case, x.size needs to be 1, which means x is a scalar, or 
           (c,) which means x is a 1-d array, or (1,c) which means 
           x is a 2-d array but just like 1-d array shape. 

    Returns
    -------
    y: baredl.Tensor
    """
    if x.shape == shape:
        return as_tensor(x)
    return BroadcastTo(shape)(x)


class ReverseBroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = self._sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

    def _sum_to(self, x, shape):
        """Sum elements along axes to output an array of a given shape.
        Args:
            x (ndarray): Input array.
            shape:
        Returns:
            ndarray: Output array of the shape.
        """
        ndim = len(shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y


def reverse_broadcast_to(x, shape):
    if x.shape == shape:
        return as_tensor(x)
    return ReverseBroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


# -------------------------------------------------------------
# max / min / clip
# -------------------------------------------------------------


class Max(Function):
    """
    Parameters
    ----------
    axis: None or int or tuple of ints
        Axis or axes along which a sum is performed. 
        If None, will sum all of the elements of the input array.

    keepdims: bool
        If True, the dimension of the result Tensor
        will be same dimension as the input Tensor. 
    """
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        """
        Parameters
        ----------
        x: np.ndarray (any shape)

        Returns
        -------
        y: np.ndarray
        """
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor

        Returns
        -------
        y: np.ndarray
        """
        x = self.inputs[0]
        y = self.outputs[0]() # weakref
        
        shape = self._backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond
    
    def _backward_shape(self, x, axis):
        if axis is None:
            axis = range(x.ndim)
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = axis

        shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
        return shape


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# -------------------------------------------------------------
# repeat_interleave
# -------------------------------------------------------------


# !!! need unit test !!!
class RepeatInterleave(Function):
    def __init__(self, repeats=2, dim=None):
        self.repeats = repeats
        self.dim = dim

    def forward(self, x):
        self.x_shape = x.shape
        xp = get_array_module(x)
        y = xp.repeat(x, self.repeats, self.dim)
        return y

    def backward(self, gy):
        if self.dim is None:
            gx = gy.reshape(-1, self.repeats).sum(axis=1).reshape(self.x_shape)
        else:
            new_shape = list(self.x_shape)
            new_shape.insert(self.dim+1,self.repeats)
            gx = gy.reshape(*new_shape).sum(axis=self.dim+1)
        return gx


def repeat_interleave(x, repeats=2, dim=None):
    return RepeatInterleave(repeats, dim)(x)