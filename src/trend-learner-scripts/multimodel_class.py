# -*- coding: utf8

from __future__ import division, print_function

from sklearn.base import clone
from sklearn.metrics import classification_report

from learn_base import create_grid_search
from learn_base import load_categories

import boosting
import cotrain
import stacking
import numpy as np
import os
import plac
import sys

def load_features(features_folder, best_by, gamma_max):
    
    F = []
    matrices = {}
    feats_fname = 'year#%d.txt'

    for i  in xrange(best_by.shape[0]):
        bby = best_by[i]

        if bby == np.inf:
            feats_file = os.path.join(features_folder, feats_fname % gamma_max)
        else:
            bby = int(bby)
            feats_file = os.path.join(features_folder, feats_fname % bby)

        if bby in matrices:
            Fi = matrices[bby]
        else:
            Fi = np.genfromtxt(feats_file)[:,1:]
            matrices[bby] = Fi

        feats = Fi[i]

        F.append(feats)

    return np.asanyarray(F)

def save_results(out_folder, base_name, y_pred, y_true):
    folder = os.path.join(out_folder, base_name)
    
    try:
        os.mkdir(folder)
    except:
        pass
    
    out_file = os.path.join(folder, 'pred.dat')
    np.savetxt(out_file, y_pred)

    with open(os.path.join(folder, 'summ.dat'), 'w') as summ_file:
        print(classification_report(y_true, y_pred), file=summ_file)

def run_classifier(out_folder, trend_probs, referrers, y, train, test):

    F = referrers #static features
    etree = create_grid_search('lr', n_jobs = 1)
    
    y_pred = trend_probs[test].argmax(axis=1)
    save_results(out_folder, 'tl-base-lr', y_pred, y[test])

    aux = clone(etree)
    aux.fit(F[train], y[train])
    y_pred = aux.predict(F[test])
    save_results(out_folder, 'tree-feats', y_pred, y[test])
    
    aux = clone(etree)
    aux.fit(trend_probs[train], y[train])
    y_pred = aux.predict(trend_probs[test])
    save_results(out_folder, 'tree-probs', y_pred, y[test])
    
    C = np.hstack((F, trend_probs))
    aux = clone(etree)
    aux.fit(C[train], y[train])
    y_pred = aux.predict(C[test])
    save_results(out_folder, 'meta-combine', y_pred, y[test])

    #stack_clf = stacking.Stacking(3, [etree], 'tree')
    #stack_clf.fit(F[train], y[train], trend_probs[train])
    #y_pred = stack_clf.predict(F[test], trend_probs[test])
    #save_results(out_folder, 'meta-stack-tree', y_pred)
    
    stack_clf = stacking.Stacking(3, [etree], 'linear')
    stack_clf.fit(F[train], y[train], trend_probs[train])
    y_pred = stack_clf.predict(F[test], trend_probs[test])
    save_results(out_folder, 'meta-stack-linear', y_pred, y[test])
    
    #stack_clf = stacking.Stacking(3, [etree], 'deco')
    #stack_clf.fit(F[train], y[train], trend_probs[train])
    #y_pred = stack_clf.predict(F[test], trend_probs[test])
    #save_results(out_folder, 'meta-stack-svm', y_pred)

def run_one_folder(features_folder, fold_folder, results_name, gamma_max):

    #File paths
    best_by_test_fpath = os.path.join(fold_folder, results_name,
            'best-by.dat')
    best_by_train_fpath = os.path.join(fold_folder, results_name + '-train',
            'best-by.dat')
    
    all_conf_test_fpath = os.path.join(fold_folder, results_name, 
            'all-conf.dat')
    all_conf_train_fpath = os.path.join(fold_folder, results_name + '-train',
            'all-conf.dat')
    
    ytest_fpath = os.path.join(fold_folder, 'ksc', 'test_assign.dat')
    ytrain_fpath = os.path.join(fold_folder, 'ksc', 'assign.dat')
    
    test_fpath = os.path.join(fold_folder, 'test.dat')
    train_fpath = os.path.join(fold_folder, 'train.dat')
    tags_fpath = os.path.join(features_folder, 'tags.dat')
    
    #Loading Matrices
    best_by_test = np.genfromtxt(best_by_test_fpath)
    best_by_train = np.genfromtxt(best_by_train_fpath)
    
    test = np.loadtxt(test_fpath, dtype='bool')
    train = np.loadtxt(train_fpath, dtype='bool')

    assert np.logical_xor(train, test).all()
    assert best_by_train.shape == train.sum()
    assert best_by_test.shape == test.sum()

    best_by = np.zeros(best_by_test.shape[0] + best_by_train.shape[0])
    best_by[test] = best_by_test
    best_by[train] = best_by_train

    trend_probs_test = np.genfromtxt(all_conf_test_fpath)
    trend_probs_train = np.genfromtxt(all_conf_train_fpath)
    
    assert trend_probs_train.shape[0] == train.sum()
    assert trend_probs_test.shape[0] == test.sum()
    
    shape = (trend_probs_test.shape[0] + trend_probs_train.shape[0], 
            trend_probs_test.shape[1])
    trend_probs = np.zeros(shape)
    trend_probs[test] = trend_probs_test
    trend_probs[train] = trend_probs_train

    y_true_test = np.loadtxt(ytest_fpath, dtype='i')
    y_true_train = np.loadtxt(ytrain_fpath, dtype='i')

    assert y_true_train.shape[0] == train.sum()
    assert y_true_test.shape[0] == test.sum()

    y_true = np.zeros(y_true_train.shape[0] + y_true_test.shape[0])
    y_true[test] = y_true_test
    y_true[train] = y_true_train
    
    referrers = load_features(features_folder, best_by, gamma_max)

    #Actual test, ufa
    run_classifier(os.path.join(fold_folder, results_name), 
            trend_probs, referrers, y_true, train, test)

@plac.annotations(
        features_folder=plac.Annotation('Folder with features', type=str),
        fold_folder=plac.Annotation('Folder with the train and test data', type=str),
        results_name=plac.Annotation('Base name of the results folder', type=str),
        gamma_max=plac.Annotation('Gamma Max', type=int))
def main(features_folder, fold_folder, results_name, gamma_max):
    run_one_folder(features_folder, fold_folder, results_name, gamma_max)

if __name__ == '__main__':
    sys.exit(plac.call(main))
