#-*- coding: utf8
'''
Common functions for creating classifiers and regressors for machine learning
tasks
'''
from __future__ import division, print_function

from sklearn import ensemble
from sklearn import grid_search
from sklearn import svm

import cStringIO
import numpy as np

#Params
SVM_C_RANGE = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
SVM_GAMMA_RANGE = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

TREE_SPLIT_RANGE = [1, 2, 4, 8, 16]

PARAMS = {'rbf_svm':{'C':SVM_C_RANGE, 'gamma':SVM_GAMMA_RANGE},
          'linear_svm':{'C':SVM_C_RANGE},
          'extra_trees':{'min_samples_split':TREE_SPLIT_RANGE}}

#Classifiers
CACHE_SIZE = 1024 * 4
CLFS = {'rbf_svm':svm.SVC(kernel='rbf', cache_size=CACHE_SIZE),
        'linear_svm':svm.LinearSVC(),
        'extra_trees':ensemble.ExtraTreesClassifier(n_estimators=20, 
                                                    compute_importances=True,
                                                    criterion='entropy')}

CLFS_SPARSE = {'rbf_svm':svm.sparse.SVC(kernel='rbf', cache_size=CACHE_SIZE),
               'linear_svm':svm.sparse.LinearSVC(),
               'extra_trees':CLFS['extra_trees']}

#Regressors
RGRS = {'rbf_svm':svm.SVR(kernel='rbf', cache_size=CACHE_SIZE),
        'linear_svm':svm.SVR(kernel='linear'),
        'extra_trees':ensemble.ExtraTreesRegressor(n_estimators=20, 
                                                   compute_importances=True)}

RGRS_SPARSE = {'rbf_svm':svm.sparse.SVR(kernel='rbf', cache_size=CACHE_SIZE),
               'linear_svm':svm.sparse.SVR(kernel='linear'),
               'extra_trees':CLFS['extra_trees']}

def _get_classifier_and_params(name, sparse = False):
    if sparse:
        dict_to_use = CLFS_SPARSE
    else:
        dict_to_use = CLFS
    
    return dict_to_use[name], PARAMS[name]

def _get_regressor_and_params(name, sparse = False):
    if sparse:
        dict_to_use = RGRS_SPARSE
    else:
        dict_to_use = RGRS
    
    return dict_to_use[name], PARAMS[name]

def create_grid_search(name, sparse=False, regressor=False, n_jobs=1):
    if regressor:
        learner, params = _get_regressor_and_params(name, sparse)
    else:
        learner, params = _get_classifier_and_params(name, sparse)
        
    return grid_search.GridSearchCV(learner, params, cv=3, refit=True, 
                                    n_jobs=n_jobs)

def load_referrers(referrers_fpath):
    X = np.genfromtxt(referrers_fpath)[:,1:]

    col_names = {}
    with open(referrers_fpath) as referrers_file:
        for line in referrers_file:
            if '#' in line:
                spl = line.split()[1:]
                col_names = dict((k, v) for k, v in enumerate(spl))

                return X, col_names

def create_input_table(referrers_fpath = None, tseries_fpath = None, 
                       num_pts = 3):

    col_names = {}
    if referrers_fpath:
        X_ref, col_names = load_referrers(referrers_fpath)
        X = X_ref
        
    if tseries_fpath and num_pts > 0:
        time_series = np.genfromtxt(tseries_fpath)[:,1:]
        X_series = time_series[:,range(num_pts)]
        X = X_series
        base = len(col_names)
        col_names.update((pnt + base, 'POINT_%d'%pnt) for pnt in range(num_pts))
        
    if referrers_fpath and tseries_fpath:
        X = np.hstack((X_ref, X_series))
    
    inverse_names = dict((v, k) for k, v in col_names.items())
    return X, col_names, inverse_names

def clf_summary(mean_scores, ci_scores):
    
    buff = cStringIO.StringIO()
    try:
        print('class \tprecision \trecall \tf1 score \tsupport', file=buff)
        for j in xrange(mean_scores.shape[1]):
            print(j, end="\t", file=buff)
            for i in xrange(mean_scores.shape[0]):
                print('%.3f +- %.3f' % (mean_scores[i, j], ci_scores[i, j]), 
                      end="\t", file=buff)
            print(file=buff)
        print(file=buff)
    
        return buff.getvalue()
    finally:
        buff.close()
