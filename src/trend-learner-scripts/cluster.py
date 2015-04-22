# -*- coding: utf8

from __future__ import division, print_function

from pyksc import dist
from pyksc import ksc

import ioutil
import numpy as np
import os
import plac

def main(tseries_fpath, base_folder, k):
    k = int(k)
    
    idx_fpath = os.path.join(os.path.join(base_folder, '..'), 'train.dat')
    X = ioutil.load_series(tseries_fpath, idx_fpath)

    cent, assign, shift, dists_cent = ksc.inc_ksc(X, k)
    np.savetxt(os.path.join(base_folder, 'cents.dat'), cent, fmt='%.5f')
    np.savetxt(os.path.join(base_folder, 'assign.dat'), assign, fmt='%d')
    np.savetxt(os.path.join(base_folder, 'shift.dat'), shift, fmt='%d')
    np.savetxt(os.path.join(base_folder, 'dists_cent.dat'), dists_cent, 
               fmt='%.5f')

if __name__ == '__main__':
    plac.call(main)
