# -*- coding: utf8

from __future__ import division, print_function

from scipy import stats

from collections import defaultdict
from matplotlib import pyplot as plt
from scripts import initialize_matplotlib

import numpy as np
import plac
import sys

cols = {'PEAK_VIEWS':3, 'SUM_VIEWS':-1}

@plac.annotations(features_fpath=plac.Annotation('Features file', type=str),
                  classes_fpath=plac.Annotation('Video classes file', type=str),
                  tseries_fpath=plac.Annotation('Time Series file', type=str))
def main(features_fpath, classes_fpath, tseries_fpath):
    X = np.genfromtxt(features_fpath)[:,1:].copy()
    y = np.loadtxt(classes_fpath)
    T  = np.genfromtxt(tseries_fpath)[:,1:].copy()

    bah = T.sum(axis=1) / X[:,-1]
    print(np.mean(bah))
    print(np.median(bah))
    print(np.std(bah))
    print(stats.scoreatpercentile(bah, 25))

    num_clusters = len(set(y))


    for k in xrange(num_clusters):
        print(k, end='\t')
        M = X[y == k]

        for column, col_num in sorted(cols.items()):
            data = M[:,col_num]
            mean = np.mean(data)
            print(mean, end='\t')
        print()

    print('Tot.', end='\t')
    for column, col_num in sorted(cols.items()):
        data = X[:,col_num]
        
        mean = np.mean(data)
        print(mean, end='\t')
    print()

if __name__ == '__main__':
    sys.exit(plac.call(main))
