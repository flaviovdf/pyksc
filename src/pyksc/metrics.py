#-*- coding: utf8

from __future__ import division, print_function

from pyksc.dist import dist_all

import numpy as np

def cost(tseries, assign, centroids, dist_centroids=None):
    
    num_series = tseries.shape[0]
    if dist_centroids is None:
        dist_centroids = dist_all(centroids, tseries)
    
    cost_f = 0.0
    for i in xrange(num_series):
        k = assign[i]
        cost_f += dist_centroids[k, i] ** 2
    
    return cost_f / num_series

def avg_intra_dist(tseries, assign, dists_all_pairs=None):
    
    num_series = tseries.shape[0]
    
    if dists_all_pairs is None:
        dists_all_pairs = dist_all(tseries, tseries, rolling=True)[0]
    
    dists = []
    for i in xrange(num_series):
        k = assign[i]
        members = assign == k
        dists_i = dists_all_pairs[i]
        dists.extend(dists_i[members])
        
    return np.mean(dists)

def avg_inter_dist(tseries, assign, dists_all_pairs=None):
    
    num_series = tseries.shape[0]
    
    if dists_all_pairs is None:
        dists_all_pairs = dist_all(tseries, tseries, rolling=True)[0]
    
    dists = []
    for i in xrange(num_series):
        k = assign[i]
        non_members = assign != k
        dists_i = dists_all_pairs[i]
        dists.extend(dists_i[non_members])
        
    return np.mean(dists)

def beta_cv(tseries, assign, dists_all_pairs=None):
    
    intra = avg_intra_dist(tseries, assign, dists_all_pairs)
    inter = avg_inter_dist(tseries, assign, dists_all_pairs)
    
    return intra / inter

def silhouette(tseries, assign, dists_all_pairs=None):
    
    if dists_all_pairs is None:
        dists_all_pairs = dist_all(tseries, tseries, rolling=True)[0]

    num_series = tseries.shape[0]
    sils = np.zeros(num_series, dtype='f')
    labels = set(assign)
    for i in xrange(num_series):
        
        k = assign[i]
        dists_i = dists_all_pairs[i]
        intra = np.mean(dists_i[assign == k])
        
        min_inter = float('inf')
        for o in labels:
            if o != k:
                inter = np.mean(dists_i[assign == o])
                if inter < min_inter:
                    min_inter = inter
         
        sils[i] = (min_inter - intra) / max(intra, min_inter)
    
    return np.mean(sils)
