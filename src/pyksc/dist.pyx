#-*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''
Basic array functions are kept here. Also, in this module
we implement the time series distance metric defined in [1].

References
----------
.. [1] J. Yang and J. Leskovec,
    "Patterns of Temporal Variation in Online Media" - WSDM'11
    http://dl.acm.org/citation.cfm?id=1935863
'''
from __future__ import division, print_function

from cpython cimport bool
from libc.stdlib cimport abort
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdio cimport printf

from cython.parallel import parallel
from cython.parallel import prange

cimport cython
cimport numpy as np
import numpy as np

np.import_array()

#Basic math functions
cdef extern from "math.h" nogil:
    double sqrt(double)

cdef extern from "cblas.h" nogil:
    double cblas_dnrm2(int N, double *X, int incX)
    double cblas_ddot(int N, double *X, int incX, double *Y, int incY)

#Inlines, some basic blas vector stuff renamed for legacy and disabling gil
cdef inline double cinner_prod(double *array1, double *array2, \
        Py_ssize_t size) nogil: \
        return cblas_ddot(size, array1, 1, array2, 1)

cdef inline double csqsum(double *array1, Py_ssize_t size) nogil: \
        return cblas_dnrm2(size, array1, 1) ** 2

cdef inline double cnorm(double *array1, Py_ssize_t size) nogil: \
        return cblas_dnrm2(size, array1, 1)

#CDEF functions
cdef double* cshift_drop(double[::1] array, int amount) nogil:
    '''
    Shifts the array by N positions. This is similar to a binary shift where
    the element's fall of at the ends.
    '''
    cdef Py_ssize_t size = array.shape[0]
    
    cdef double *shifted
    shifted = <double *> malloc(size * sizeof(double))
    if shifted == NULL:
        abort()

    cdef Py_ssize_t delta_shifted = 0
    cdef Py_ssize_t delta_array = 0
    if amount > 0:
        delta_shifted = amount
    else:
        delta_array = -amount
        amount = -amount

    cdef Py_ssize_t i = 0
    for i in range(size):
        shifted[i] = 0

    i = 0
    for i in range(size - amount):
        shifted[i + delta_shifted] = array[i + delta_array]
    
    return shifted

cdef double* cshift_roll(double[::1] array, int amount) nogil:
    '''
    Shifts the array by N positions. This is a rolling shifts, where elements 
    come back at the other side of the array.
    '''
    cdef Py_ssize_t size = array.shape[0]

    cdef Py_ssize_t delta_shifted = 0
    cdef Py_ssize_t delta_array = 0
    if amount > 0:
        delta_shifted = amount
    else:
        delta_array = -amount

    cdef double *shifted
    shifted = <double *> malloc(size * sizeof(double))
    if shifted == NULL:
        abort()

    cdef Py_ssize_t i = 0
    for i in range(size):
        shifted[(i + delta_shifted) % size] = array[(i + delta_array) % size]

    return shifted

cdef double cshift_dist(double[::1] array1, double[::1] array2, 
                        int shift_amount, int rolling) nogil:
    '''
    Computes the distance between two time series using a given shift.
    '''
    cdef Py_ssize_t size = array1.shape[0]
    if size == 0:
        return 0
    
    cdef double *shifted
    if rolling:
        shifted = cshift_roll(array2, shift_amount)
    else:
        shifted = cshift_drop(array2, shift_amount)
    
    #computing scaling
    cdef double alpha
    cdef double sqsum_shift = csqsum(shifted, size)
    if sqsum_shift != 0:
        alpha = cinner_prod(&array1[0], shifted, size) / sqsum_shift
    else:
        alpha = 0

    #actual distance
    cdef Py_ssize_t i = 0
    cdef double dist = 0
    for i in range(size):
        dist += (array1[i] - alpha * shifted[i]) ** 2
    
    free(shifted)
        
    cdef double norm1 = cnorm(&array1[0], size)
    if norm1 != 0:
        return sqrt(dist) / norm1
    elif sqsum_shift != 0: #array one is all zeros, but 2 is not
        return 1
    else: #both are all zeros
        return 0

cdef ds_pair_t* cdist(double[::1] array1, double[::1] array2, int rolling) nogil:
    '''
    Computes the distance between two time series by searching for the optimal
    shifting parameter.
    '''

    cdef Py_ssize_t size = array1.shape[0]
    cdef ds_pair_t* rv = <ds_pair_t*>malloc(sizeof(ds_pair_t))
    if rv == NULL:
        abort()

    if size == 0:
        rv.min_dist = 0
        rv.best_shift = 0
        return rv

    cdef double distance
    cdef double best_distance = 1
    cdef Py_ssize_t best_shift = 0

    cdef Py_ssize_t i
    for i in range(-size + 1, size):
        distance = cshift_dist(array1, array2, i, rolling)
        if distance < best_distance:
            best_distance = distance
            best_shift = i

    rv.min_dist = best_distance
    rv.best_shift = best_shift
    return rv

cdef tuple cdist_all(double[:, ::1] matrix1, double[:, ::1] matrix2, int rolling):
    '''
    Computes the distance between all pairs of rows in the given matrices.
    The elements of the first matrix are the ones which will be shifted.
    '''
    
    cdef Py_ssize_t n_rows1 = matrix1.shape[0]
    cdef Py_ssize_t n_rows2 = matrix2.shape[0]
    cdef Py_ssize_t n_cols = matrix1.shape[1]
    
    cdef np.ndarray[double, ndim=2] rv_dist = np.ndarray((n_rows1, n_rows2))
    cdef np.ndarray[int, ndim=2] rv_shifts = np.ndarray((n_rows1, n_rows2),
            dtype='i')

    cdef ds_pair_t*** aux = <ds_pair_t***> malloc(n_rows1 * sizeof(ds_pair_t**))
    if aux == NULL:
        abort()

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in prange(n_rows1, nogil=True, schedule='static'):
        aux[i] = <ds_pair_t**> malloc(n_rows2 * sizeof(ds_pair_t*))
        if aux[i] == NULL:
            abort()

        for j in range(n_rows2):
            aux[i][j] = cdist(matrix1[i], matrix2[j], rolling)
            rv_dist[i, j] = aux[i][j].min_dist
            rv_shifts[i, j] = aux[i][j].best_shift

            free(aux[i][j])
        free(aux[i])
    free(aux)

    return (rv_dist, rv_shifts)

#Python wrappers
def shift(np.ndarray[double, ndim=1, mode='c'] array not None, int amount,
          bool rolling=False):
    '''
    Shifts the array by N positions. This is a rolling shifts, where elements 
    come back at the other side of the array. This method return a new array,
    it does not do inplace shifts.

    Arguments
    ---------
    array: np.ndarray[np.float_t, ndim=1]
        The array to shift
    amount: int
        The amount to shuft by, positive integer signal right shifts while
        negative ones signal left shifts
    rolling: bool (default `False`)
        indicates whether we should use a rolling distance (i.e. elements at
        one end re appear at another) or a drop distance (i.e. elements fall
        and zeroes take their place, similar to a binary shift)
    '''

    cdef Py_ssize_t size = array.shape[0]
    cdef double *shift_buff
    if rolling:
        shift_buff = cshift_roll(array, amount)
    else:
        shift_buff = cshift_drop(array, amount)

    cdef np.ndarray[double, ndim=1] rv = np.ndarray(size)
    free(rv.data)
    rv.data = <char *>shift_buff
    return rv

def inner_prod(np.ndarray[double, ndim=1, mode='c'] array1 not None,
               np.ndarray[double, ndim=1, mode='c'] array2 not None):
    '''
    Return's the inner product between two arrays. It is a necessity for both 
    arrays to have the same shape.
    
    Arguments
    ---------
    array1: np.ndarray[np.float_t, ndim=1]
        First array
    array2: np.ndarray[np.float_t, ndim=1]
        Second array
    '''

    assert array1.shape[0] == array2.shape[0]
    cdef Py_ssize_t size = array1.shape[0]
    return cinner_prod(&array1[0], &array2[0], size)

def sqsum(np.ndarray[double, ndim=1, mode='c'] array not None):
    '''
    Returns the squared sum of the elements in the given array.

    Arguments
    ---------
    array: np.ndarray[np.float_t, ndim=1]
        The array to sum the elements
    '''
    
    return csqsum(&array[0], array.shape[0])

def shift_dist(np.ndarray[double, ndim=1, mode='c'] array1 not None,
               np.ndarray[double, ndim=1, mode='c'] array2 not None, 
               int shift_amount, bool rolling=False):
    '''
    Computes the distance between two time series. This is an implementation
    of the distance metric define in Section 2.2 of [1]. This is the distance
    metric for a fixed shifting parmeter, where the scaling can be easily
    computed.

    Arguments
    ---------
    array1: np.ndarray[np.float_t, ndim=1]
        First time series
    array2: np.ndarray[np.float_t, ndim=1]
        Second time series
    shift_amout: int
        the shifting parameter
    rolling: bool (default `False`)
        indicates whether we should use a rolling distance (i.e. elements at
        one end reappear at another) or a drop distance (i.e. elements fall
        and zeroes take their place, similar to a binary shift)
    
    References
    ----------
    .. [1] J. Yang and J. Leskovec,
       "Patterns of Temporal Variation in Online Media" - WSDM'11
        http://dl.acm.org/citation.cfm?id=1935863
    '''
    assert array1.shape[0] == array2.shape[0]
    
    if rolling:
        return cshift_dist(array1, array2, shift_amount, 1)
    else:
        return cshift_dist(array1, array2, shift_amount, 0)

def dist(np.ndarray[double, ndim=1, mode='c'] array1 not None, 
         np.ndarray[double, ndim=1, mode='c'] array2 not None,
         bool rolling=False):
    '''
    Computes the distance between two time series. This is an implementation
    of the distance metric define in Section 2.2 of [1]. It searchs for optimal
    scaling and shifting paramters to align both series and compare similarity
    mostly based on *shape*. 
    
    This is a symmetric measure *only* when using rolling shifts.

    Arguments
    ---------
    array1: np.ndarray[np.float_t, ndim=1, mode='c']
        First time series
    array2: np.ndarray[np.float_t, ndim=1, mode='c']
        Second time series
    rolling: bool (default `False`)
        indicates whether we should use a rolling distance (i.e. elements at
        one end reappear at another) or a drop distance (i.e. elements fall
        and zeroes take their place, similar to a binary shift)

    References
    ----------
    .. [1] J. Yang and J. Leskovec, 
       "Patterns of Temporal Variation in Online Media" - WSDM'11  
        http://dl.acm.org/citation.cfm?id=1935863
    '''
    assert array1.shape[0] == array2.shape[0]

    cdef ds_pair_t* rv
    cdef int roll = 0
    if rolling:
        roll = 1

    try:
        rv = cdist(array1, array2, roll)
        return rv.min_dist
    finally:
        free(rv)

def dist_all(np.ndarray[double, ndim=2, mode='c'] matrix1 not None,
             np.ndarray[double, ndim=2, mode='c'] matrix2 not None,
             bool rolling=False):

    '''
    Computes the distance between all of examples (rows) from the first
    matrix to all other examples in the second matrix. The return value
    is a matrix of n_rows1, n_rows2 containing the distances.

    The elements of the first matrix are the ones which will be shifted.

    Both matrices must have the same number of columns.

    Arguments
    ---------
    matrix1: np.ndarray[np.float_t, ndim=2, mode='c']
        A matrix of time series
    matrix2: np.ndarray[np.float_t, ndim=2, mode='c']
        A matrix of time series
    rolling: bool (default `False`)
        indicates whether we should use a rolling distance (i.e. elements at
        one end reappear at another) or a drop distance (i.e. elements fall
        and zeroes take their place, similar to a binary shift)
    '''

    assert matrix1.shape[1] == matrix2.shape[1]
    cdef int roll = 0
    if rolling:
        roll = 1

    return cdist_all(matrix1, matrix2, roll)
