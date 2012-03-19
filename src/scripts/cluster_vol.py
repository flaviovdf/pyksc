# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict
from matplotlib import pyplot as plt
from scripts import initialize_matplotlib

import numpy as np
import plac
import sys

cols = {'PEAK_DATE_FRACTION':-6, 'PEAK_VIEWS_FRACTION':-4, 
        'PEAK_VIEWS_TOTAL':-3, 'SUM_VIEWS':-2}

@plac.annotations(features_fpath=plac.Annotation('Features file', type=str),
                  classes_fpath=plac.Annotation('Video classes file', type=str))
def main(features_fpath, classes_fpath):
    initialize_matplotlib()

    X = np.genfromtxt(features_fpath)[:,1:].copy()
    y = np.loadtxt(classes_fpath)

    num_clusters = len(set(y))

    print(end='\t')
    for column in sorted(cols):
        print(column, end='\t')
    print()

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
