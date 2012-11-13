#-*- coding: utf8

#A basic structure for returning a pair
cdef struct ds_pair_t:
    double min_dist
    int best_shift

#Distance function
cdef ds_pair_t* cdist(double[:] array1, double[:] array2, int rolling) nogil
