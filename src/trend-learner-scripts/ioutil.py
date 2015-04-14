#-*- coding: utf8

import numpy as np

EPS = 1e-6

def load_series(tseries_fpath, idx_fpath):
    usecols = range(1, 101)
    X = np.genfromtxt(tseries_fpath, usecols = usecols) + EPS
    train_idx = np.loadtxt(idx_fpath, dtype='bool')
    return np.asanyarray(X[train_idx], order='C')
