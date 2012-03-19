#-*- coding: utf8
'''
Implementation of the KSC and IncrementalKSC algorithms.  See [1] for details.
Both algorithms can be used for clustering time series data, the second
(IncrementalKSC) being an optimization of the initial clusters heuristic to
be used by the first.

References
----------
.. [1] J. Yang and J. Leskovec, 
   "Patterns of Temporal Variation in Online Media" - WSDM'11  
   http://dl.acm.org/citation.cfm?id=1935863
'''
from __future__ import division, print_function

from pyksc.dhwt import transform
from pyksc.dist import dist_all
from pyksc.dist import shift

from pyksc.metrics import cost

import numpy as np
import numpy.linalg as LA

def _compute_centroids(tseries, assign, num_clusters, to_shift=None):
    '''
    Given a time series matrix and cluster assignments, this method will
    compute the spectral centroids for each cluster.
    
    Arguments
    ---------
    tseries: matrix (n_series, n_points)
        Time series beng clustered
    assign: array of ints (size = n_series)
        The cluster assignment for each time series
    num_clusters: int
        The number of clusters being searched for
    to_shift (optional): array of ints (size = n_series)
        Determines if time series should be shifted, if different from `None`.
        In this case, each series will be shifted by the corresponding amount
        in the array.
    '''

    series_size = tseries.shape[1]
    centroids = np.ndarray((num_clusters, series_size))

    #shift series for best centroid distance
    #TODO: this method can be cythonized and done in parallel
    shifted = tseries
    if to_shift is not None:
        for i in xrange(tseries.shape[0]):
            shifted[i] = shift(tseries[i], to_shift[i], rolling=True)

    #compute centroids
    for k in xrange(num_clusters):
        members = shifted[assign == k]
        if members.any():
            num_members = 0
            if members.ndim == 2:
                axis = 1
                num_members = members.shape[0]
            else:
                axis = 0
                num_members = 1
            
            ssqs = np.tile(np.sum(members**2, axis=axis), (series_size, 1))
            #the original papers divides by ssqs only, while the author's
            #example implementation uses sqrt. We chose sqrt because it appears
            #to yield better centroids.
            aux = members / np.sqrt(ssqs.T)

            x_mat = np.dot(aux.T, aux)
            i_mat = num_members * np.eye(series_size)
            m_mat = i_mat - x_mat

            #compute eigenvalues and chose the vector for the smallest one
            #TODO: Check if using scipy's linalg is faster (has more options
            #      such as finding only the smallest eigval)
            eig_vals, eig_vectors = LA.eigh(m_mat)
            centroids[k] = eig_vectors[:,eig_vals.argmin()]
        else:
            centroids[k] = np.zeros(series_size)

    return centroids

def _base_ksc(tseries, initial_centroids, n_iters=-1):
    '''
    This is the base of the KSC algorithm. It follows the same idea of a K-Means
    algorithm. Firstly, we assign time series to a new cluster based on the
    distance to the centroids. For each time series, it is computed the best
    shift to minimize the distance to the closest centroid.
     
    The assignment step is followed by an update step where new centroids are 
    computed based on the new clustering (based on the update step).
    
    Both steps above are repeated `n_iters` times. If this parameter is negative
    then the steps are repeated until convergence, that is, until no time series
    changes cluster between consecutive steps. 

    Arguments
    ---------
    tseries: a matrix of shape (number of time series, size of each series)
        The time series to cluster
    initial_centroids: a matrix of shape (num. of clusters, size of time series)
        The initial centroid estimates
    n_iters: int
        The number of iterations which the algorithm will run

    Returns
    -------
    centroids: a matrix of shape (num. of clusters, size of time series)
        The final centroids found by the algorithm
    assign: an array of num. series size
        The cluster id which each time series belongs to
    best_shift: an array of num. series size
        The amount shift amount performed for each time series
    cent_dists: a matrix of shape (num. centroids, num. series)
        The distance of each centroid to each time serie

    References
    ----------
    .. [1] Wikipedia, 
    "K-means clustering"  
    http://en.wikipedia.org/wiki/K-means_clustering
    '''
    
    num_clusters = initial_centroids.shape[0]
    num_series = tseries.shape[0]

    centroids = initial_centroids

    #KSC algorithm
    cent_dists = None
    assign = None
    prev_assign = None
    best_shift = None

    iters = n_iters
    converged = False

    while iters != 0 and not converged:
        #assign elements to new clusters
        cent_dists, shifts = dist_all(centroids, tseries, rolling=True)
        
        assign = cent_dists.argmin(axis=0)
        best_shift = np.ndarray(num_series, dtype='i')
        for i in xrange(shifts.shape[1]):
            best_shift[i] = shifts[assign[i], i]
        
        #check if converged, if not compute new centroids
        if prev_assign is not None and not (prev_assign - assign).any():
            converged = True
        else: 
            centroids = _compute_centroids(tseries, assign, num_clusters, 
                                          best_shift)

        prev_assign = assign
        iters -= 1
    
    return centroids, assign, best_shift, cent_dists

def _bestcost_ksc(tseries, num_clusters, n_iters=-1, n_runs=10):
    
    
    min_cost = float('+inf')
    
    best_cents = None
    best_assign = None
    best_shift = None
    best_dist = None

    for i in xrange(n_runs):
        assign = np.random.randint(0, num_clusters, tseries.shape[0])
        cents = _compute_centroids(tseries, assign, num_clusters)

        cents, assign, series_shift, dists = _base_ksc(tseries, cents, n_iters)
        clust_cost = cost(tseries, assign, cents, dists)

        if clust_cost < min_cost:
            min_cost = clust_cost
            best_cents = cents
            best_assign = assign
            best_shift = series_shift
            best_dist = dists

    return best_cents, best_assign, best_shift, best_dist

def ksc(tseries, num_clusters, n_iters=-1, n_runs=10):
    return _bestcost_ksc(tseries, num_clusters, n_iters, n_runs)

def inc_ksc(tseries, num_clusters, n_iters=-1, num_wavelets=2):
    
    dhw_series = []
    dhw_series.append(tseries)
    previous = tseries
    for i in xrange(num_wavelets):
        new_series = []
        for j in xrange(tseries.shape[0]):
            wave = transform(previous[j])[0]
            new_series.append(wave)

        previous = np.array(new_series)
        dhw_series.append(previous)

    assign = np.random.randint(0, num_clusters, tseries.shape[0])
    cents = None
    series_shift = None
    for dhw in reversed(dhw_series):
        cents = _compute_centroids(dhw, assign, num_clusters, series_shift)
        cents, assign, series_shift, dists = _base_ksc(dhw, cents, n_iters)
    
    return cents, assign, series_shift, dists
