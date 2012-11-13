#-*- coding: utf8

from cython.parallell import prange
from cython.view cimport array as cvarray

from libc.stdlib cimport free

from pyksc cimport dist

cimport cython
cimport numpy as np

np.import_array()

#Basic math functions
cdef extern from "math.h" nogil:
    float min(float, float)

cdef extern from "math.h" nogil:
    float exp(float)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float dist_to_reference(float[:] s, float[:] r) nogil:
    cdef Py_ssize_t n_obs = s.shape[0]
    cdef Py_ssize_t n_ref = r.shape[0]

    cdef float min_dist = 1
    cdef Py_ssize_t i
    cdef dist.ds_pair_t* dist_rv
    for i in range(n_ref - n_obs + 1):
        dist_rv = dist.cdist(r[i:i + n_obs], s, n_obs, 1)
        min_dist = min(min_dist, dist_rv.min_dist)
        free(dist_rv)

    return min_dist

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void predict_one(float[:] s, float[:, ::1] R_pos,
        float gamma, float theta, int num_steps, float[:] store_at) nogil:

    cdef Py_ssize_t num_windows = s.shape[0] + 1
    cdef Py_ssize_t num_pos = R_pos.shape[0]

    cdef int num_detections = 0
    cdef float[:] new_s
    cdef double prob
    cdef Py_ssize_t i
    cdef Py_ssize_t j

    for i from 1 <= i < num_windows:
        new_s = s[:i]

        prob = 0
        for j from 0 <= j < num_pos:
            prob += exp(-gamma * dist_to_reference(new_s, R_pos[j]))

        if prob >= theta and num_steps >= i:
            store_at[0] = prob
            store_at[1] = i
            break

@cython.boundscheck(False)
@cython.wraparound(False)
def predict(np.ndarray[np.float_t, ndim=2, mode='c'] X not None, 
            np.ndarray[np.float_t, ndim=2, mode='c'] R not None, 
            np.ndarray[np.float_t, ndim=1, mode='c'] labels not None,
            float gamma, float theta, int num_steps, int num_threads):

    cdef Py_ssize_t num_samples = X.shape[0]
    cdef Py_ssize_t num_points = X.shape[1]

    cdef Py_ssize_t num_labels = labels.shape[0]

    cdef float[:] s
    cdef float[:, ::1] R_pos
    cdef float[:, ::1] return_val = \
            np.zeros(shape=(num_samples, 2), dtype=np.float, order='C')

    cdef Py_ssize_t i = 0
    for l from 0 <= l < num_labels:
        #TODO: Maybe this copy is not necessary, need to check.
        R_pos = np.asanyarray(R[labels == l], dtype=np.float, order='C')

        for i in prange(num_samples, schedule='static', num_threads=num_threads):
            predict_one(X[i], R_pos, gamma, theta, num_steps, return_val[i])

    return return_val
