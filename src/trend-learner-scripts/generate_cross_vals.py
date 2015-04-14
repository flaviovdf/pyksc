#!/usr/bin/env python
# -*- coding: utf8
from __future__ import print_function, division

from sklearn import cross_validation

import numpy as np
import os
import plac
import sys

def main(tseries_fpath, out_folder):
    X = np.genfromtxt(tseries_fpath)[:,1:]
    num_series = X.shape[0]
    
    curr_fold = 1
    cv = cross_validation.KFold(num_series, 5, indices=False)
    for train, test in cv:
        curr_out_folder = os.path.join(out_folder, 'fold-%d' % curr_fold)
        
        try:
            os.makedirs(curr_out_folder)
        except:
            pass

        np.savetxt(os.path.join(curr_out_folder, 'train.dat'), train, fmt='%i')
        np.savetxt(os.path.join(curr_out_folder, 'test.dat'), test, fmt='%i')
        curr_fold += 1

if __name__ == '__main__':
    sys.exit(plac.call(main))
