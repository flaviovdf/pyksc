# -*- coding: utf8

from __future__ import division, print_function

from scipy.stats.mstats import mquantiles

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

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

def pred(probs_folder, num_series, max_pts, min_pts, thetas):
    
    y_pred = np.zeros(num_series) - 1
    best_by = np.zeros(num_series) + np.inf
    confs = np.zeros(num_series)
    all_confs = np.zeros((num_series, len(thetas)))

    for num_pts in range(1, max_pts + 1):
        fpath = os.path.join(probs_folder, FNAME) % num_pts
        P = np.loadtxt(fpath)
        
        curr_pred = P.argmax(axis=1)
        curr_score = P.max(axis=1)

        for i in xrange(num_series):
            score = curr_score[i]
            curr_cls = curr_pred[i]
            theta = thetas[curr_cls]
            min_req = min_pts[curr_cls]

            if num_pts >= min_req and score > theta and y_pred[i] == -1:
                y_pred[i] = curr_cls
                best_by[i] = num_pts
                confs[i] = score
                all_confs[i] = P[i]

                #if y_pred[i] != curr_cls and score > confs[i]:
                #    y_pred[i] = curr_cls
                #    #best_by[i] = num_pts
                #    confs[i] = score
                #    all_confs[i] = P[i]

    assert y_pred[confs > 0].sum() == y_pred[y_pred != -1].sum()
    assert y_pred[best_by != np.inf].sum() == y_pred[y_pred != -1].sum()
    
    return y_pred, best_by, confs, all_confs

def aux_print(X, peak_days, sum_views, best_by, y_true, y_pred, confs, 
        idx, summ_file):

    X = X[idx]
    peak_days = peak_days[idx]
    sum_views = sum_views[idx]
    best_by = np.asanyarray(best_by[idx], dtype='i')
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    confs = confs[idx]

    left_frac = np.zeros(X.shape[0])
    for i in xrange(X.shape[0]):
        left_frac[i] = \
                (sum_views[i] - X[i][:best_by[i]].sum()) / sum_views[i]
    
    dist_peak = (peak_days - best_by - 1)

    print('- PeakDistQuantiles (peak - best)', mquantiles(dist_peak), file=summ_file)
    print('- LeftViewsQuantiles', mquantiles(left_frac), file=summ_file)


def save_results(X, peak_days, sum_views, pts_grid, theta_grid, best_by, all_confs,
        y_true, y_pred, confs, out_folder):

    valid = confs > 0
    correct = y_true == y_pred

    summ_fpath = os.path.join(out_folder, 'summ.dat')
    with open(summ_fpath, 'w') as summ_file:
        print('Params', file=summ_file)
        for cls in sorted(pts_grid):
            print('\t Cls = %d; min_pts = %d; theta = %.3f' \
                    % (cls, pts_grid[cls], theta_grid[cls]), file=summ_file)
        print(file=summ_file)

        print('All', file=summ_file)
        aux_print(X, peak_days, sum_views, best_by, y_true, y_pred, confs, valid, summ_file)
        print(file=summ_file)
        
        print('Correct Only', file=summ_file)
        aux_print(X, peak_days, sum_views, best_by, y_true, y_pred, confs, valid & correct, summ_file)
        print(file=summ_file)
        
        print('Incorrect Only', file=summ_file)
        aux_print(X, peak_days, sum_views, best_by, y_true, y_pred, confs, valid & ~correct, summ_file)
        print(file=summ_file)

        print(classification_report(y_true[valid], y_pred[valid]), 
                file=summ_file)
        print(file=summ_file)
        print('# invalid %d' % (~valid).sum(), file=summ_file)

    ypred_fpath = os.path.join(out_folder, 'pred.dat')
    np.savetxt(ypred_fpath, y_pred)

    bestby_fpath = os.path.join(out_folder, 'best-by.dat')
    np.savetxt(bestby_fpath, best_by)

    conf_fpath = os.path.join(out_folder, 'conf.dat')
    np.savetxt(conf_fpath, confs)
    
    conf_fpath = os.path.join(out_folder, 'all-conf.dat')
    np.savetxt(conf_fpath, all_confs)

def run_fold(folder, tseries_fpath, min_pts, thetas, out_folder):

    try:
        os.makedirs(out_folder)
    except:
        pass

    test_fpath = os.path.join(folder, 'test.dat')
    cents_fpath = os.path.join(folder, 'ksc', 'cents.dat')
    assign_fpath = os.path.join(folder, 'ksc', 'test_assign.dat')
    probs_folder = os.path.join(folder, 'probs-test')

    X = ioutil.load_series(tseries_fpath, test_fpath)
    test_idx = np.loadtxt(test_fpath, dtype='bool')
    y_true = np.loadtxt(assign_fpath)
    
    num_series = X.shape[0]
    max_pts = X.shape[1]
    
    #Since we prune the first 100 lines of X we need to read other info
    peak_days = []
    sum_views = []
    with open(tseries_fpath) as tseries_file:
        for i, line in enumerate(tseries_file):
            if test_idx[i]:
                x = np.array([int(v) for v in line.split()[1:]])
                peak_days.append(x.argmax())
                sum_views.append(x.sum())

    peak_days = np.array(peak_days)
    sum_views = np.array(sum_views)
  
    y_pred, best_by, confs, all_confs = \
            pred(probs_folder, num_series, max_pts, min_pts, thetas)
    save_results(X, peak_days, sum_views, min_pts, thetas, best_by, all_confs,
                 y_true, y_pred, confs, out_folder)


def get_params(folder, threshold):
    
    assign = np.loadtxt(os.path.join(folder, 'ksc', 'assign.dat'), dtype='i')
    P = np.loadtxt(os.path.join(folder, 'probs', 'all-conf.dat'), dtype='f')
    best_by = np.loadtxt(os.path.join(folder, 'probs', 'best-by.dat'), dtype='i')
    
    thetas = {}
    min_pts = {}
    for i in xrange(2, P.shape[1]):
        fpath = os.path.join(folder, 'probs', 'probs-%d-pts.dat' % i)
        Pi = np.loadtxt(fpath, dtype='f')
        for k in set(assign):
            y_true = assign == k

            maxcls = Pi.argmax(axis=1)
            y_pred = maxcls == k
            score = f1_score(y_true, y_pred)
            if score >= threshold and k not in thetas:
                thetas[k] = P[assign == k][:,i].mean()
                min_pts[k] = i

    return thetas, min_pts

def multi_proc_run(args):

    folder, tseries_fpath = args
    fitted_thetas, fitted_min_pts = get_params(folder, .5)

    out_folder = os.path.join(folder, 'cls-res-fitted-50')
    run_fold(folder, tseries_fpath, fitted_min_pts, fitted_thetas, 
                out_folder)

def main(tseries_fpath, base_folder):
     
    folders = glob.glob(os.path.join(base_folder, 'fold-*/'))
    pool = multiprocessing.Pool()
    pool.map(multi_proc_run, [(fold, tseries_fpath) for fold in folders])
    pool.terminate()
    pool.join()

if __name__ == '__main__':
    sys.exit(plac.call(main))
