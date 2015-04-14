#!/usr/bin/env python
# -*- coding: utf8
from __future__ import print_function, division

from pyksc import dist

import glob
import numpy as np
import os
import plac
import sys

def main(tseries_fpath, in_folder):

    ids = []
    with open(tseries_fpath) as tseries_file:
        for l in tseries_file:
            ids.append(l.split()[0])

    ids = np.array(ids)
    folders = glob.glob(os.path.join(in_folder, 'fold-*/ksc'))
    num_folders = len(folders)

    agree = 0
    diff = 0
    
    for i in xrange(num_folders):

        base_i = os.path.dirname(folders[i])
        Ci = np.loadtxt(os.path.join(folders[i], 'cents.dat'))

        train_i = np.loadtxt(os.path.join(base_i, 'train.dat'), dtype='bool')
        assign_i = np.loadtxt(os.path.join(folders[i], 'assign.dat'))

        for j in xrange(i, num_folders):

            base_j = os.path.dirname(folders[j])    
            Cj = np.loadtxt(os.path.join(folders[j], 'cents.dat'))
            
            dists = dist.dist_all(Ci, Cj, rolling=True)[0]
            argsrt = dists.argsort(axis=1)
            
            train_j = np.loadtxt(os.path.join(base_j, 'train.dat'), dtype='bool')    
            assign_j = np.loadtxt(os.path.join(folders[j], 'assign.dat'))
            
            for k in xrange(argsrt.shape[0]):
                first = True
                for o in argsrt[k]:
                    ids_k = set(ids[train_i][assign_i == k])
                    ids_o = set(ids[train_j][assign_j == o])
                    n_inter = len(ids_k.intersection(ids_o))

                    if first:
                        first = False
                        agree += n_inter
                    else:
                        diff += n_inter
    
    print('AgreedProb = ', agree / (agree + diff))
    print('DisagreeProb = ', diff / (agree + diff))

if __name__ == '__main__':
    sys.exit(plac.call(main))
