# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 02:15:09 2017

@author: Etcyl 
         Using signaltools by Oliphant to run the Wiener function with only the necessary code.
         Ver. 1.0 Testing
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

#Create some test data
data = [1.19E-06,
0.017992973,
0.018664837,
0.018156052,
0.01938796,
0.017963886,
0.020537138,
0.020867109,
0.018296957,
0.018646955,
0.018135071,
0.02096796,
0.018605947,
0.017978907,
0.018259287,
0.018981934,
0.016915798,
0.018337011,
0.018617153,
0.018064976,
0.018497944,
0.019276142,
0.017889023,
0.017562866,
0.018094063,
0.019376755,
0.022032261,
0.024926901,
0.023641825,
0.017882109,
0.021186113,
0.020604849,
0.019976139,
0.018776894,
0.017739058,
0.01867795,
0.018745899,
0.020033121,
0.018153906,
0.024551153,
0.015916109,
0.018162012,
0.020334005,
0.021250725,
0.018448114,
0.019503832,
0.017515182,
0.018023014,
0.019212008,
0.020375013,
0.020177126,
0.019347906,
0.021155119,
0.017460823,
0.021929979,
0.019533873,
0.01687336,
0.020342588,
0.018561125,
0.015971184,
0.020376921,
0.017390966,
0.020749092,
0.018577099,
0.017547846,
0.018460989,
0.019042969,
0.02098918,
0.017118931,
0.019558907,
0.019517899,
0.018195152,
0.020522833,
0.018691063,
0.021212101,
0.016366959,
0.019822121,
0.019961119,
0.017756939,
0.017903805,
0.01730299,
0.018580914,
0.018376112,
0.017479181,
0.02132678,
0.019622087,
0.01865983,
0.018347979,
0.017833948,
0.018035412,
0.020501852,
0.01668787,
0.018800974,
0.01931119,
0.022050858,
0.023372173,
0.019279957,
0.0227561,
0.02220273,
0.01913023,
0.019098759,
0.018759251,
0.022236824,
0.018593073,
0.021599054,
0.020282745,
0.018730164,
0.01932621,
0.022636652,
0.018224239,
0.017753839,
0.01792717,
0.01826787,
0.0175879,
0.020195007,
0.022525072,
0.020837069,
0.018754005,
0.02312088,
0.018112183,
0.020706892,
0.018850803,
0.019641161,
0.020781994,
0.021370888,
0.018694162,
0.019109964,
0.018705845,
0.018584013,
0.018782139,
0.019566059,
0.021554947,
0.019831896,
0.017086029,
0.01713109,
0.017447948,
0.018913031,
0.018265963,
0.018876076,
0.020667076,
0.02172184,
0.016839266,
0.019845724,
0.018870115,
0.019438982,
0.018668175,
0.019788742,
0.017923117,
0.01808691,
0.018545151,
0.016959906,
0.016782999,
0.016256809,
0.018926144,
0.022916794,
0.023168087,
0.016242266,
0.018944025,
0.018002748,
0.017812014,
0.017987251,
0.021061659,
0.017840147,
0.018574953,
0.02128315,
0.016811848,
0.018425226,
0.016974926,
0.018511057,
0.020234823,
0.018064022,
0.017559052,
0.017100096,
0.022382975,
0.016790152,
0.01872468,
0.018676043]

#Generate an output from the Wiener Filter (1x)
output = wiener(data)

#Filter the data again (2x)
output2 = wiener(output)
