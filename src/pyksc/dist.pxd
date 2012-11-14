#-*- coding: utf8

#A basic structure for returning a pair
cdef struct ds_pair_t:
    double min_dist
    int best_shift

#Distance function
cdef ds_pair_t* cdist(double[::1] array1, double[::1] array2, int rolling) nogil

cdef double cshift_dist(double[::1] array1, double[::1] array2, \
        int shift_amount, int rolling) nogil
