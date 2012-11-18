#-*- coding: utf8

#A basic structure for the return value of the distance func
cdef struct dist_struct_t:
    double dist
    double alpha
    int shift

#Distance function
cdef dist_struct_t* cdist(double[::1] array1, double[::1] array2, int rolling)\
        nogil

cdef dist_struct_t* cshift_dist(double[::1] array1, double[::1] array2,\
        int shift_amount, int rolling) nogil

