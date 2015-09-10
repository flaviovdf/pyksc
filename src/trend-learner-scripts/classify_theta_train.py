# -*- coding: utf8

from __future__ import division, print_function

from classify_theta import get_params, pred, save_results

from pyksc import dist

import argparse
import glob
import ioutil
import multiprocessing
import numpy as np
import os
import plac
import sys

FNAME = 'probs-%d-pts.dat'

def run_fold(folder, tseries_fpath, min_pts, thetas, gamma_max, out_folder):

    try:
        os.makedirs(out_folder)
    except:
        pass

    train_fpath = os.path.join(folder, 'train.dat')
    cents_fpath = os.path.join(folder, 'ksc', 'cents.dat')
    assign_fpath = os.path.join(folder, 'ksc', 'assign.dat')
    probs_folder = os.path.join(folder, 'probs')

    X = ioutil.load_series(tseries_fpath, train_fpath)
    train_idx = np.loadtxt(train_fpath, dtype='bool')
    y_true = np.loadtxt(assign_fpath)
    
    num_series = X.shape[0]
    max_pts = gamma_max
    
    #Since we prune the first 100 lines of X we need to read other info
    peak_days = []
    sum_views = []
    with open(tseries_fpath) as tseries_file:
        for i, line in enumerate(tseries_file):
            if train_idx[i]:
                x = np.array([int(v) for v in line.split()[1:]])
                peak_days.append(x.argmax())
                sum_views.append(x.sum())

    peak_days = np.array(peak_days)
    sum_views = np.array(sum_views)
  
    y_pred, best_by, confs, all_confs = \
            pred(probs_folder, num_series, max_pts, min_pts, thetas)
    save_results(X, peak_days, sum_views, min_pts, thetas, best_by, all_confs,
                 y_true, y_pred, confs, out_folder)

def multi_proc_run(args):

    folder, tseries_fpath, f1_target, gamma_max, results_sub_folder, max_k = args
    fitted_thetas, fitted_min_pts = get_params(folder, f1_target, max_k)

    out_folder = os.path.join(folder, results_sub_folder)
    run_fold(folder, tseries_fpath, fitted_min_pts, fitted_thetas, 
                gamma_max, out_folder)

def main(tseries_fpath, base_folder, f1_target, results_sub_folder, gamma_max, max_k):
    gamma_max = int(gamma_max)
    max_k = int(max_k)
    
    f1_target = float(f1_target)
    folders = glob.glob(os.path.join(base_folder, 'fold-*/'))
    pool = multiprocessing.Pool()
    pool.map(multi_proc_run, [(fold, tseries_fpath, f1_target, gamma_max, results_sub_folder, max_k) for fold in folders])
    pool.terminate()
    pool.join()

if __name__ == '__main__':
    sys.exit(plac.call(main))
