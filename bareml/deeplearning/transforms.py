from abc import ABCMeta, abstractmethod
import numpy as np
from .core import as_tensor
from .utils import pair

# PIL is not in dependency list
try:
    from PIL import Image
except ImportError:
    Image = None


# -------------------------------------------------------------
# Compose & Base classes
# -------------------------------------------------------------


class Compose:
    """ 
    Compose multiple transforms 
    
    Parameters
    ----------
    transforms: list of Transform instances
    """
    
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data):
        """
        Apply transforms

        Parameters
        ----------
        data: ndarray (any shape)
        """
        if not self.transforms:
            return data
        
        for f in self.transforms:
            data = f(data)
        return data


class Transform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, batch):
        """
        Parameters
        ----------
        batch: list of objects (n,) or np.array (n, *dim)
            n: number of samples in a batch
            if it's list, element can be anything such as PIL image.
            if it's np.ndarray, *dim depends on each dataset. 
        """
        pass


class TransformPIL(Transform):

    def __init__(self):
        if Image is None:
            raise Exception('Image is not available. Install PIL.')


# -------------------------------------------------------------
# Transforms for np.ndarray
# -------------------------------------------------------------


class Flatten(Transform):
    def __call__(self, batch):
        """
        batch: np.ndarray(n, *dim)
            n: number of samples in a batch
            *dim: depends on dataset
        """
        return np.array([array.flatten() for array in batch])


class AsType(Transform):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, batch):
        """
        batch: np.ndarray(n, *dim)
            n: number of samples in a batch
            *dim: depends on dataset
        """
        return batch.astype(self.dtype)


class ToInt(AsType):
    def __init__(self):
        self.dtype = np.int


class ToFloat(AsType):
    def __init__(self):
        self.dtype = np.float32


class ToTensor(Transform):
    def __call__(self, batch):
        """
        Make the entire batch as a tensor

        Parameters
        ----------
        batch: np.ndarray(n, *dim)
            n: number of samples in a batch
            *dim: depends on dataset
        """
        return as_tensor(batch)


class Normalise(Transform):
    """Normalize a NumPy array with mean and standard deviation.
    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence):
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, batch):
        """
        batch: np.ndarray(n, *dim)
            n: number of samples in a batch
            *dim: depends on dataset
        """
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * batch.ndim
            mshape[1] = batch.shape[1] if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=batch.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * batch.ndim
            rshape[1] = batch.shape[1] if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=batch.dtype).reshape(*rshape)
        return (batch - mean) / std


# -------------------------------------------------------------
# Transforms for PIL Image
# -------------------------------------------------------------


class ToArray(TransformPIL):
    """Convert PIL Image to NumPy array."""
    def __init__(self, dtype=np.float32):
        super().__init__()
        self.dtype = dtype

    def _to_array(self, img):
        img = np.asarray(img)
        if img.ndim == 2: # gray scale
            img = np.array([img]) # convert from (H,W) to (1,H,W)
        else:
            img = img.transpose(2, 0, 1) # convert from (H,W,C) to (C,H,W)
        img = img.astype(self.dtype)
        return img

    def __call__(self, batch):
        """
        batch: list of PIL Image (n,)
            n: number of samples in a batch
        """
        if isinstance(batch, np.ndarray):
            return batch
        if isinstance(batch[0], Image.Image):
            return np.array([self._to_array(img) for img in batch])
        else:
            raise TypeError


class ToPIL(TransformPIL):
    """
    Convert NumPy array to PIL Image.
    https://stackoverflow.com/questions/51479140/convert-numpy-array-object-to-pil-image-object
    """

    def _to_PIL(self, array):
        if array.shape[0] == 1: # gray scale
          data = array[0] # convert to (H,W) shape
        else: # RGB scale
          data = array.transpose(1, 2, 0) # convert to (H,W,C) shape
        return Image.fromarray(data)

    def __call__(self, batch):
        """
        batch: np.ndarray(n, c, h, w)
            n: number of samples in a batch
            c: number of channels
            h: height
            w: width
        """
        return [self._to_PIL(array) for array in batch]


class Resize(TransformPIL):
    """Resize the input PIL image to the given size.
    Args:
        size (int or (int, int)): Desired output size
        mode (int): Desired interpolation.
    """
    def __init__(self, size, mode=Image.BILINEAR):
        super().__init__()
        self.size = pair(size)
        self.mode = mode

    def __call__(self, batch):
        """
        batch: list of PIL Image (n,)
            n: number of samples in a batch
        """
        return [img.resize(self.size, self.mode) for img in batch]