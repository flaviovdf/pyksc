# -*- coding: utf8

from __future__ import division, print_function

from matplotlib import dates
from matplotlib import pyplot as plt

from scripts import initialize_matplotlib

import argparse
import numpy as np
import os
import sys
import traceback

refs = {
'G_EXTERNAL_EVENT_DATE':0, 
'G_FEATURED_EVENT_DATE':1, 
'G_INTERNAL_EVENT_DATE':2,
'G_MOBILE_EVENT_DATE':3, 
'G_SEARCH_EVENT_DATE':4, 
'G_SOCIAL_EVENT_DATE':5, 
'G_VIRAL_EVENT_DATE':6
}

UP_DATE = -1

def main(features_fpath):
    initialize_matplotlib()
    
    X = np.genfromtxt(features_fpath)[:,1:]
    
    for r, k in sorted(refs.items()):
        idxs = X[:,k] > 0
        time_to_ref = (X[:,UP_DATE][idxs] - X[:,k][idxs])
        print(r, np.mean(time_to_ref), np.std(time_to_ref))

    print('peak_frac', np.mean(X[:,-3]), np.std(X[:,-3]))
    
    time_to_peak = (X[:,-4] - X[:,UP_DATE]) / 7
    print('peak_date', np.mean(time_to_peak), np.std(time_to_peak))
    
    import time
    plt.hist(X[:,UP_DATE], bins=20)
    ticks, labels = plt.xticks()
    plt.xticks(ticks, [time.strftime('%m/%y', time.localtime(x)) for x in ticks])
    plt.ylabel('\# Videos')
    plt.xlabel('Month/Year')
    plt.savefig('hist.pdf')
    
    
#        plt.plot(t_series, '-k')
#        plt.ylabel('Views')
#        plt.xlabel('Time')
#        plt.savefig(os.path.join(plot_foldpath, '%d.pdf' % i))
#        plt.close()
#        
#        half = t_series.shape[0] // 2
#        to_shift = half - np.argmax(t_series)
#        to_plot_peak_center = dist.shift(t_series, to_shift, rolling=True)
#        plt.plot(to_plot_peak_center, '-k')
#        plt.ylabel('Views')
#        plt.xlabel('Time')
#        plt.savefig(os.path.join(plot_foldpath, '%d-peak-center.pdf' % i))
#        plt.close()
#        
#        to_shift = 0 - np.argmin(t_series)
#        to_plot_min_first = dist.shift(t_series, to_shift, rolling=True)
#        plt.plot(to_plot_min_first, '-k')
#        plt.ylabel('Views')
#        plt.xlabel('Time')
#        plt.savefig(os.path.join(plot_foldpath, '%d-min-first.pdf' % i))
#        plt.close()
#        
#    np.savetxt(os.path.join(plot_foldpath, 'cents.dat'), cent, fmt='%.5f')
#    np.savetxt(os.path.join(plot_foldpath, 'assign.dat'), assign, fmt='%d')
#    np.savetxt(os.path.join(plot_foldpath, 'shift.dat'), shift, fmt='%d')
#    np.savetxt(os.path.join(plot_foldpath, 'dists_cent.dat'), dists_cent, 
#               fmt='%.5f')

def create_parser(prog_name):
    
    desc = __doc__
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog_name, description=desc,
                                     formatter_class=formatter)
    
    parser.add_argument('features_fpath', type=str, help='Features file')
    
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.features_fpath)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))