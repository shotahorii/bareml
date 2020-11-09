"""
Helper functions.
"""

import numpy as np
from .core import Tensor, get_array_module, cupy


# -------------------------------------------------------------
# General helper functions: logsumexp, pair
# -------------------------------------------------------------


def logsumexp(x, axis=1):
    """
    https://blog.feedly.com/tricks-of-the-trade-logsumexp/

    Parameters
    ----------
    x: np.ndarray (n, c)
        n: number of samples
        c: number of classes

    axis: int

    Returns
    -------
    m: np.ndarray (n,1)
    """
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple) and len(x) == 2:
        return x
    else:
        raise ValueError


# -------------------------------------------------------------
# Helper functions for unit test: numerical_diff 
# -------------------------------------------------------------


def numerical_diff(f, x, eps=1e-4):
    """
    Parameters
    ----------
    f: function
    x: baredl.Tensor

    Returns
    -------
    y: xp.ndarray 
    """
    x0 = Tensor(x.data - eps)
    x1 = Tensor(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# -------------------------------------------------------------
# Helper functions for conv: 
# -------------------------------------------------------------


def get_conv_outsize(in_size, ker_size, stride, pad):
    """
    Calculate output size of conv operation.
    Kalculate for either height or width each. 

    Parameters
    ----------
    in_size: int    input size
    ker_size: int   kernel size
    stride: int     stride
    pad: int        padding
    """
    return (in_size + pad * 2 - ker_size) // stride + 1


def get_deconv_outsize(in_size, ker_size, stride, pad):
    """
    Calculate output size of transpose conv operation.
    Kalculate for either height or width each. 

    Parameters
    ----------
    in_size: int    input size
    ker_size: int   kernel size
    stride: int     stride
    pad: int        padding
    """
    return stride * (in_size - 1) + ker_size - 2 * pad


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    """
    Purpose of this function is to convert an image (N, C, H, Width)
    into the shape that is tensordot-able with W (OC, C, KH, KW).
    The output shape would be (N, C, KH, KW, OH, OW), which is 
    tensordot-able in (1,2,3) axis. 
    https://www.youtube.com/watch?v=PWPJVws7l0M&feature=emb_title (in JP)
    
    Parameters
    ----------
    img: xp.array (N, C, H, Width)
        N: number of samples (images)
        C: number of channels
        H: height of the images
        Width: width of the images
        e.g. img with shape (3,2,3,4) is like below.
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

    kernel_size: int K or tuple of 2 ints (KH, KW)
        KH: height of the kernel
        KW: width of the kernel
        if the input is an int K, KH = KW = K

    stride: int or tuple of 2 ints    stride

    pad: int or tuple of 2 ints       padding

    Returns
    -------
    col: xp.ndarray (N, C, KH, KW, OH, OW)

    """

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH) # output height 
    OW = get_conv_outsize(W, KW, SW, PW) # output width

    xp = get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        # the pad width of the first 2 dimensions are both (0, 0), which 
        # means no padding applied to those dimensions = samples & channels.
        # the padding applied to the last 2 dimensions = img height & width.
        pad_width = ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1))
        
        # pad_fill specifies only 1 value here. 
        # this means this value (0,) applies to all dimensions. 
        pad_fill = (0,)

        """
        np.pad(...) adds padding in the given img. 
        Let's assume SH = SW = PH = PW = 1, which means 
        pad_width = ((0, 0), (0, 0), (1, 1), (1, 1)), 
        and the input img is below array (3,2,3,4).
            np.array([
                # sample1
                [
                    [[1,1,1,1],[1,1,1,1],[1,1,1,1]], # channel1 (3*4 matrix)
                    [[1,1,1,1],[1,1,1,1],[1,1,1,1]]  # channel2 (3*4 matrix)
                ],
                # sample2
                [
                    [[1,1,1,1],[1,1,1,1],[1,1,1,1]], # channel1 (3*4 matrix)
                    [[1,1,1,1],[1,1,1,1],[1,1,1,1]]  # channel2 (3*4 matrix)
                ],
                # sample 3
                [
                    [[1,1,1,1],[1,1,1,1],[1,1,1,1]], # channel1 (3*4 matrix)
                    [[1,1,1,1],[1,1,1,1],[1,1,1,1]]  # channel2 (3*4 matrix)
                ],
            ])
        Then, the output would be below:
            np.array([
                # sample1
                [
                    [[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]], # channel1 (5*6 matrix)
                    [[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]]  # channel2 (5*6 matrix)
                ],
                # sample2
                [
                    [[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]], # channel1 (5*6 matrix)
                    [[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]]  # channel2 (5*6 matrix)
                ],
                # sample 3
                [
                    [[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]], # channel1 (5*6 matrix)
                    [[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]]  # channel2 (5*6 matrix)
                ],
            ])
        """
        img = np.pad(img, pad_width, mode='constant', constant_values=pad_fill)

        """
        Make an empty container of 6-dim array (Note: below [[*]] = an empty OH*OW matrix)
        Assuming KH = KW = 2, then
            np.array([
                # sample1
                [
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ], # channel1 (KH*KW matrix of OH*OW matrix)
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ]  # channel2 (KH*KW matrix of OH*OW matrix)
                ],
                # sample2
                [ 
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ], # channel1 (KH*KW matrix of OH*OW matrix)
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ]  # channel2 (KH*KW matrix of OH*OW matrix)
                ],
                # sample 3
                [
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ], # channel1 (KH*KW matrix of OH*OW matrix)
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ]  # channel2 (KH*KW matrix of OH*OW matrix)
                ],
            ])
        """
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        # Note that, OH is the height of the output image, but 
        # at the same time, OH also means the number of kernel 
        # application to the input image in height (vertical) direction. 
        # This is same for OW. 
        #
        # [j, i] represents each pixel of the kernel.
        # j represents the position of vertical direction. 
        # for each j in KH, j is the first position in img to sweep through,
        # and j_lim = j + SH * OH is the end position in img after the sweep. 
        # This is same for i and i_lim = i + SW * OW.
        #
        # So, j:j_lim:SH, i:i_lim:SW represents all the positions in the img
        # where the kernel's pixcel [j, i] goes through. 
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    """
    Purpose of this function is to convert a "col" array (N, C, KH, KW, OH, OW)
    into the shape of its original image (N, C, H, W).
    Can be considered as a reverse function of im2col_array above, but 
    note that col2im(im2col(img)) != img in general, because col2im adds 
    values in overlapped kernel areas multiple times.

    Parameters
    ----------
    col: xp.ndarray (N, C, KH, KW, OH, OW)
        e.g. Assuming N = 3, C=2, KH = KW = 2, then below. (Note: below [[*]] = an empty OH*OW matrix)
            np.array([
                # sample1
                [
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ], # channel1 (KH*KW matrix of OH*OW matrix)
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ]  # channel2 (KH*KW matrix of OH*OW matrix)
                ],
                # sample2
                [ 
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ], # channel1 (KH*KW matrix of OH*OW matrix)
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ]  # channel2 (KH*KW matrix of OH*OW matrix)
                ],
                # sample 3
                [
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ], # channel1 (KH*KW matrix of OH*OW matrix)
                    [ [[[*]], [[*]]], [[[*]], [[*]]] ]  # channel2 (KH*KW matrix of OH*OW matrix)
                ],
            ])

    img_shape: tuple (N, C, H, W)
        N: number of samples
        C: number of channels
        H: height of the image
        W: width of the image

    kernel_size: int K or tuple of 2 ints (KH, KW)
        KH: height of the kernel
        KW: width of the kernel
        if the input is an int K, KH = KW = K

    stride: int or tuple of 2 ints    stride

    pad: int or tuple of 2 ints       padding

    Returns
    -------
    img: xp.array (N, C, H, W)
    """
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]

        # exclude padding area of the image
        img_without_pad = img[:, :, PH:H + PH, PW:W + PW]
        return img_without_pad


def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cupy.empty((n, c, h, w), dtype=col.dtype)

    cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img

