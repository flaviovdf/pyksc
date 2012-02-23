#-*- coding: utf8
'''
Implements Discrete Harr Wavelet Transform (also inverse) for a time series.
This is simply done by computing the average of consecutive elements in the
vector that correspond to the time series. See [1] and [2] for details.

References
----------
.. [1] P. Van Fleet
   "The Discrete Haar Wavelet Tranformation"
   http://goo.gl/IPz25
   (last access December 2011)
   
.. [2] I. Kaplan
   "Applying the Haar Wavelet Transform to Time Series Information"
   http://www.bearcave.com/misl/misl_tech/wavelets/haar.html
   (last access December 2011)
'''
from __future__ import division, print_function

cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def transform(np.ndarray[np.float_t, ndim=1] array):
    '''
    Transform the array to a new form using the discrete harr
    transform operation. This is computing the average of consecutive 
    elements in the array.

    Arguments
    ---------
    array: np.ndarray[np.float_t, ndim=1]
        the array to transform

    Returns
    -------
    This method returns a tuple being the first element the wavelet and
    the second the coefficients to be used to transform the wavelet back
    to the original array.
    '''
    cdef Py_ssize_t n = array.shape[0]
    cdef Py_ssize_t new_dim

    if n % 2 == 0:
        new_dim = n // 2
    else:
        new_dim = (n // 2) + 1

    cdef np.ndarray[np.float_t, ndim=1] wavelet = np.zeros(new_dim)
    cdef np.ndarray[np.float_t, ndim=1] coefficient = np.zeros(new_dim)

    cdef float first
    cdef float second
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0

    for i in range(0, n, 2):
        first = array[i]
        if i < n - 1:
            second = array[i + 1]
        else:
            second = 0
        
        wavelet[j] = (first + second) / 2
        coefficient[j] = (first - second) / 2
        j += 1

    return wavelet, coefficient

@cython.boundscheck(False)
@cython.wraparound(False)
def inverse(np.ndarray[np.float_t, ndim=1] wavelet,
            np.ndarray[np.float_t, ndim=1] coefficient):
    '''
    Given a wavelet and its coefficients this method can be used to 
    transform the wavelet to the original array.
    
    Arguments
    ---------
    wavelet: np.ndarray[np.float_t, ndim=1]
             the wavelet to transform back
    coefficient: np.ndarray[np.float_t, ndim=1]
             the coefficients needed for the transform
    '''
    cdef Py_ssize_t n = wavelet.shape[0]

    #sanity check
    if n != coefficient.shape[0]:
        return None

    cdef Py_ssize_t new_dim
    if n % 2 == 0 or n == 1:
        new_dim = n * 2
    else:
        new_dim = n * 2 - 1

    cdef np.ndarray[np.float_t, ndim=1] array = np.zeros(new_dim)

    cdef float first
    cdef float second
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    for i in range(n):
        first = wavelet[i] + coefficient[i]
        second = wavelet[i] - coefficient[i]
        
        if j < new_dim: 
            array[j] = first

        if j + 1 < new_dim:
            array[j + 1] = second

        j += 2

    return array
