#-*- coding: utf8
# cython: boundscheck = False
# cython: wraparound = False

from cython.parallel import prange
from cython.view cimport array as cvarray

from libc.stdlib cimport free
from libc.stdio cimport printf

from pyksc cimport dist

cimport cython
cimport numpy as np

import numpy as np
np.import_array()

#Basic math functions
cdef extern from "math.h" nogil:
    double exp(double)

cdef inline double dmin(double a, double b) nogil: return a if a < b else b

cdef double dist_to_reference(double[::1] s, double[::1] r) nogil:
    cdef Py_ssize_t n_obs = s.shape[0]
    cdef Py_ssize_t n_ref = r.shape[0]

    cdef double min_dist = 1
    cdef Py_ssize_t i
    cdef double d
    
    for i in range(n_ref - n_obs + 1):
        d = dist.cshift_dist(r[i:i + n_obs], s, 0, 1)
        min_dist = dmin(min_dist, d)

    return min_dist

cdef void predict_one(double[::1] s, double[:, ::1] R_pos,
        double gamma, int num_steps, double[:, ::1] probs, 
        int store_at_row, int store_at_col) nogil:

    cdef Py_ssize_t num_windows = s.shape[0] + 1
    cdef Py_ssize_t num_pos = R_pos.shape[0]

    cdef int num_detections = 0
    cdef double[::1] new_s = s[:num_steps]
    
    cdef double prob = 0
    cdef Py_ssize_t i = 0
    for i in range(num_pos):
        prob += exp(-gamma * dist_to_reference(new_s, R_pos[i]))

    probs[store_at_row, store_at_col] = prob

def predict(np.ndarray[double, ndim=2, mode='c'] X not None, 
            np.ndarray[double, ndim=2, mode='c'] R not None, 
            np.ndarray[long, ndim=1, mode='c'] labels not None,
            int num_labels, double gamma, int num_steps):

    cdef Py_ssize_t num_samples = X.shape[0]
    cdef Py_ssize_t num_points = X.shape[1]

    cdef double[::1] s
    cdef double[:, ::1] R_pos
    
    cdef double[:, ::1] probs = \
            np.zeros(shape=(num_samples, num_labels), dtype=np.float64, 
                     order='C')
            
    cdef double[:, ::1] Xview = X #For nogil

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t l = 0
    for l from 0 <= l < num_labels:
        #TODO: Maybe this copy is not necessary, need to check.
        R_pos = np.asanyarray(R[labels == l], dtype=np.float64, order='C')

        for i in prange(num_samples, schedule='static', nogil=True):
            predict_one(Xview[i], R_pos, gamma, num_steps, probs, i, l)

    return probs.base
