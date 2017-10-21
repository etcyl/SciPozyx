# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:08:31 2017

@author: Etcyl 
         Using signaltools by Oliphant to run the Wiener function with only the necessary code.
"""
# Author: Travis Oliphant
# 1999 -- 2002

from __future__ import division, print_function, absolute_import

import warnings

from scipy.signal import sigtools
from numpy import (asarray, ones, product, mean, where, ravel)
import numpy as np

_modedict = {'valid': 0, 'same': 1, 'full': 2}
_boundarydict = {'fill': 0, 'pad': 0, 'wrap': 2, 'circular': 2, 'symm': 1,
                 'symmetric': 1, 'reflect': 4}
                 
def _valfrommode(mode):
    try:
        val = _modedict[mode]
    except KeyError:
        if mode not in [0, 1, 2]:
            raise ValueError("Acceptable mode flags are 'valid' (0),"
                             " 'same' (1), or 'full' (2).")
        val = mode
    return val

def _check_valid_mode_shapes(shape1, shape2):
    for d1, d2 in zip(shape1, shape2):
        if not d1 >= d2:
            raise ValueError(
                "in1 should have at least as many items as in2 in "
                "every dimension for 'valid' mode.")

def _bvalfromboundary(boundary):
    try:
        val = _boundarydict[boundary] << 2
    except KeyError:
        if val not in [0, 1, 2]:
            raise ValueError("Acceptable boundary flags are 'fill', 'wrap'"
                             " (or 'circular'), \n  and 'symm'"
                             " (or 'symmetric').")
        val = boundary << 2
    return val

def wiener(im, mysize=None, noise=None):
    """
    Perform a Wiener filter on an N-dimensional array.
    Apply a Wiener filter to the N-dimensional array `im`.
    Parameters
    ----------
    im : ndarray
        An N-dimensional array.
    mysize : int or arraylike, optional
        A scalar or an N-length list giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
        If mysize is a scalar, then this scalar is used as the size
        in each dimension.
    noise : float, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.
    Returns
    -------
    out : ndarray
        Wiener filtered result with the same shape as `im`.
    """
    im = asarray(im)
    if mysize is None:
        mysize = [3] * len(im.shape)
    mysize = asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), im.ndim)

    # Estimate the local mean
    lMean = correlate(im, ones(mysize), 'same') / product(mysize, axis=0)

    # Estimate the local variance
    lVar = (correlate(im ** 2, ones(mysize), 'same') / product(mysize, axis=0)
            - lMean ** 2)

    # Estimate the noise power if needed.
    if noise is None:
        noise = mean(ravel(lVar), axis=0)

    res = (im - lMean)
    res *= (1 - noise / lVar)
    res += lMean
    out = where(lVar < noise, lMean, res)

    return out


def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """
    Convolve two 2-dimensional arrays.
    Convolve `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.
    Parameters
    ----------
    in1, in2 : array_like
        Two-dimensional input arrays to be convolved.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:
        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.
    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.
    Returns
    -------
    out : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    """
    in1 = asarray(in1)
    in2 = asarray(in2)

    if mode == 'valid':
        _check_valid_mode_shapes(in1.shape, in2.shape)

    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.ComplexWarning)
        # FIXME: some cast generates a warning here
        out = sigtools._convolve2d(in1, in2, 1, val, bval, fillvalue)

    return out  
    
    
def correlate(in1, in2, mode='full'):
    """
    Cross-correlate two N-dimensional arrays.
    Cross-correlate `in1` and `in2`, with the output size determined by the
    `mode` argument.
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    Returns
    -------
    correlate : array
        An N-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.
    Notes
    -----
    The correlation z of two d-dimensional arrays x and y is defined as:
      z[...,k,...] = sum[..., i_l, ...]
                         x[..., i_l,...] * conj(y[..., i_l + k,...])
    """
    in1 = asarray(in1)
    in2 = asarray(in2)

    _modedict = {'valid': 0, 'same': 1, 'full': 2}
    # Don't use _valfrommode, since correlate should not accept numeric modes
    try:
        val = _modedict[mode]
    except KeyError:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

    if in1.ndim == in2.ndim == 0:
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    if mode == 'valid':
        _check_valid_mode_shapes(in1.shape, in2.shape)
        ps = [i - j + 1 for i, j in zip(in1.shape, in2.shape)]
        out = np.empty(ps, in1.dtype)

        z = sigtools._correlateND(in1, in2, out, val)
    else:
        ps = [i + j - 1 for i, j in zip(in1.shape, in2.shape)]
        # zero pad input
        in1zpadded = np.zeros(ps, in1.dtype)
        sc = [slice(0, i) for i in in1.shape]
        in1zpadded[sc] = in1.copy()

        if mode == 'full':
            out = np.empty(ps, in1.dtype)
        elif mode == 'same':
            out = np.empty(in1.shape, in1.dtype)

        z = sigtools._correlateND(in1zpadded, in2, out, val)

    return z      
