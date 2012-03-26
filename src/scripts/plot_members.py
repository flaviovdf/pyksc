# -*- coding: utf8

from __future__ import division, print_function

from matplotlib import pyplot as plt

from pyksc import dist
from scripts import initialize_matplotlib

import plac
import numpy as np
import os
import sys
        
def plot_series(t_series, plot_foldpath, name, shift=False):
    
    to_plot = t_series
    if shift:
        to_shift = 0 - np.argmin(t_series)
        to_plot = dist.shift(t_series, to_shift, rolling=True)
        
    plt.plot(to_plot, '-k')
    plt.ylabel('Views')
    plt.xlabel('Time')
    plt.savefig(os.path.join(plot_foldpath, '%s.png' % name))
    plt.close()

@plac.annotations(tseries_fpath=plac.Annotation('Input file', type=str),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str),
                  centroids_fpath=plac.Annotation('Cluster centroids file',
                                                  type=str),
                  plot_foldpath=plac.Annotation('Output folder', type=str)) 
def main(tseries_fpath, assign_fpath, centroids_fpath, plot_foldpath):
    initialize_matplotlib()
    
    X = np.genfromtxt(tseries_fpath)[:,1:].copy()
    y = np.genfromtxt(assign_fpath)
    centroids = np.genfromtxt(centroids_fpath)

    num_classes = len(set(y))
    
    for k in xrange(num_classes):
        centroid_plot_foldpath = os.path.join(plot_foldpath, str(k))
        os.mkdir(centroid_plot_foldpath)

        centroid = centroids[k]
        plot_series(centroid, centroid_plot_foldpath, 'centroid', True)
        
        members = X[y == k]
        n_samples = members.shape[0]
        sample_rows = np.arange(n_samples)
        np.random.shuffle(sample_rows)        
        
        members_to_plot = members[sample_rows[:10]]
        for i in xrange(members_to_plot.shape[0]):
            print(k, i)
            plot_series(members_to_plot[i], centroid_plot_foldpath, 'ex-%d' % i)
            
if __name__ == '__main__':
    sys.exit(plac.call(main))