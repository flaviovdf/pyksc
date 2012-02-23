# -*- coding: utf8

from __future__ import division, print_function

from matplotlib import pyplot as plt

from pyksc import dist
from pyksc import ksc

from scripts import initialize_matplotlib

import argparse
import numpy as np
import os
import sys
import traceback

def main(tseries_fpath, k, plot_foldpath):
    initialize_matplotlib()
    
    X = np.genfromtxt(tseries_fpath)[:,1:].copy()

    cent, assign, shift, dists_cent = ksc.inc_ksc(X, k)
    
    for i in xrange(cent.shape[0]):
        t_series = cent[i]
        
        plt.plot(t_series, '-k')
        plt.ylabel('Views')
        plt.xlabel('Time')
        plt.savefig(os.path.join(plot_foldpath, '%d.pdf' % i))
        plt.close()
        
        half = t_series.shape[0] // 2
        to_shift = half - np.argmax(t_series)
        to_plot_peak_center = dist.shift(t_series, to_shift, rolling=True)
        plt.plot(to_plot_peak_center, '-k')
        plt.ylabel('Views')
        plt.xlabel('Time')
        plt.savefig(os.path.join(plot_foldpath, '%d-peak-center.pdf' % i))
        plt.close()
        
        to_shift = 0 - np.argmin(t_series)
        to_plot_min_first = dist.shift(t_series, to_shift, rolling=True)
        plt.plot(to_plot_min_first, '-k')
        plt.ylabel('Views')
        plt.xlabel('Time')
        plt.savefig(os.path.join(plot_foldpath, '%d-min-first.pdf' % i))
        plt.close()
        
    np.savetxt(os.path.join(plot_foldpath, 'cents.dat'), cent, fmt='%.5f')
    np.savetxt(os.path.join(plot_foldpath, 'assign.dat'), assign, fmt='%d')
    np.savetxt(os.path.join(plot_foldpath, 'shift.dat'), shift, fmt='%d')
    np.savetxt(os.path.join(plot_foldpath, 'dists_cent.dat'), dists_cent, 
               fmt='%.5f')

def create_parser(prog_name):
    
    desc = __doc__
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog_name, description=desc,
                                     formatter_class=formatter)
    
    parser.add_argument('tseries_fpath', type=str, help='Time series file')
    parser.add_argument('k', type=int, help='Number of clusters')
    parser.add_argument('plot_foldpath', type=str, help='Folder to store plots')
    
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.tseries_fpath, values.k, values.plot_foldpath)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))