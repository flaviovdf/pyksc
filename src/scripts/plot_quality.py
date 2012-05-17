# -*- coding: utf8

from __future__ import division, print_function

from matplotlib import pyplot as plt

from pyksc import dist
from pyksc import metrics
from pyksc import ksc

from scripts import initialize_matplotlib

from vod.stats.ci import half_confidence_interval_size as hci

import argparse
import numpy as np
import os
import sys
import traceback

def run_clustering(X, k, dists_all):

    cent, assign, shift, dists_cent = ksc.inc_ksc(X, k)

    intra = metrics.avg_intra_dist(X, assign, dists_all)[0]
    inter = metrics.avg_inter_dist(X, assign, dists_all)[0]
    bcv = metrics.beta_cv(X, assign, dists_all)
    cost = metrics.cost(X, assign, None, dists_cent)

    return intra, inter, bcv, cost
    
def main(tseries_fpath, plot_foldpath):
    assert os.path.isdir(plot_foldpath)
    initialize_matplotlib()
    
    X = np.genfromtxt(tseries_fpath)[:,1:].copy()

    n_samples = X.shape[0]
    sample_rows = np.arange(n_samples)
    
    clust_range = range(2, 16)
    n_clustering_vals = len(clust_range)
    
    intra_array = np.zeros(shape=(25, n_clustering_vals))
    inter_array = np.zeros(shape=(25, n_clustering_vals))
    bcvs_array = np.zeros(shape=(25, n_clustering_vals))
    costs_array = np.zeros(shape=(25, n_clustering_vals))
    
    r = 0
    for i in xrange(5):
        np.random.shuffle(sample_rows)
        rand_sample = sample_rows[:200]
        
        X_new = X[rand_sample]
        D_new = dist.dist_all(X_new, X_new, rolling=True)[0]
        
        for j in xrange(5):
            for k in clust_range:
                intra, inter, bcv, cost = run_clustering(X_new, k, D_new)
                
                intra_array[r, k - 2] = intra
                inter_array[r, k - 2] = inter
                bcvs_array[r, k - 2]  = bcv
                costs_array[r, k - 2] = cost
                
            r += 1
            print(r)

    intra_err = np.zeros(n_clustering_vals)
    inter_err = np.zeros(n_clustering_vals)
    bcvs_err = np.zeros(n_clustering_vals)
    costs_err = np.zeros(n_clustering_vals)

    for k in clust_range:
        j = k - 2
        intra_err[j] = hci(intra_array[:,j], .95)
        inter_err[j] = hci(inter_array[:,j], .95)
        bcvs_err[j] = hci(bcvs_array[:,j], .95)
        costs_err[j] = hci(costs_array[:,j], .95)
            
    plt.errorbar(clust_range, np.mean(inter_array, axis=0), fmt='gD', 
                 label='Inter Cluster', yerr=inter_err)
    plt.errorbar(clust_range, np.mean(bcvs_array, axis=0), fmt='bo', 
                 label='BetaCV', yerr=bcvs_err)
    plt.errorbar(clust_range, np.mean(intra_array, axis=0), fmt='rs', 
                 label='Intra Cluster', yerr=intra_err)
    plt.ylabel('Average Distance')
    plt.xlabel('Number of clusters')
    plt.xlim((0., 16))
    plt.ylim((0., 1.))
    plt.legend(frameon=False, loc='lower left')
    plt.savefig(os.path.join(plot_foldpath, 'bcv.pdf'))
    plt.close()
    
    plt.errorbar(clust_range, np.mean(costs_array, axis=0), fmt='bo', 
                 label='Cost', yerr=costs_err)
    plt.ylabel('Cost (F)')
    plt.xlabel('Number of clusters')
    plt.xlim((0., 16))
    plt.ylim((0., 1.))
    plt.legend(frameon=False, loc='lower left')
    plt.savefig(os.path.join(plot_foldpath, 'cost.pdf'))
    plt.close()

def create_parser(prog_name):
    
    desc = __doc__
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog_name, description=desc,
                                     formatter_class=formatter)
    
    parser.add_argument('tseries_fpath', type=str, help='Time series file')
    parser.add_argument('plot_foldpath', type=str, help='Folder to store plots')
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.tseries_fpath, values.plot_foldpath)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))
