#-*- coding: utf8
'''
Common functions for creating classifiers and regressors for machine learning
tasks
'''
from __future__ import division, print_function

from scripts.col_to_cluster import CATEG_ABBRV

from scipy import sparse

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
                                                    criterion='gini',
                                                    n_jobs=1)}

CLFS_SPARSE = {'rbf_svm':svm.sparse.SVC(kernel='rbf', cache_size=CACHE_SIZE),
               'linear_svm':svm.sparse.LinearSVC(),
               'extra_trees':CLFS['extra_trees']}

#Regressors
RGRS = {'rbf_svm':svm.SVR(kernel='rbf', cache_size=CACHE_SIZE),
        'linear_svm':svm.SVR(kernel='linear'),
        'extra_trees':ensemble.ExtraTreesRegressor(n_estimators=20, 
                                                   compute_importances=True,
                                                   n_jobs=1)}

RGRS_SPARSE = {'rbf_svm':svm.sparse.SVR(kernel='rbf', cache_size=CACHE_SIZE),
               'linear_svm':svm.sparse.SVR(kernel='linear'),
               'extra_trees':CLFS['extra_trees']}

#Category Parsing Utilities
CAT_COL = 2
CAT_IDS = dict((abbrv, i) \
               for i, abbrv in enumerate(sorted(set(CATEG_ABBRV.values()))))

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

def hstack_if_possible(X, Y):
    if X is not None:
        return np.hstack((X, Y))
    else:
        return Y
    
def update_col_ids(ids_to_insert, column_ids=None):
    if not column_ids: 
        column_ids = {}
        
    base = len(column_ids)
    column_ids.update((pnt + base, name) for pnt, name in ids_to_insert.items())

    return column_ids

def load_referrers(referrers_fpath, X = None, column_ids=None):
    X_ref = np.genfromtxt(referrers_fpath)[:,1:].copy()

    new_col_ids = {}
    with open(referrers_fpath) as referrers_file:
        for line in referrers_file:
            if '#' in line:
                spl = line.split()[1:]
                new_col_ids = dict((k, v) for k, v in enumerate(spl))

                return hstack_if_possible(X, X_ref), \
                    update_col_ids(new_col_ids, column_ids)

def load_time_series(tseries_fpath, num_pts = 3, X = None, column_ids=None):
    X_series = np.genfromtxt(tseries_fpath)[:,1:][:,range(num_pts)]
    
    new_col_ids = dict((i, 'POINT_%d'%pnt) \
                       for i, pnt in enumerate(range(num_pts)))

    return hstack_if_possible(X, X_series), \
        update_col_ids(new_col_ids, column_ids)
                      
def load_categories(tags_cat_fpath, X = None, column_ids=None):
    with open(tags_cat_fpath) as tags_cat_file:
        data = []
        row = []
        col = []
        new_col_ids = {}
        for i, line in enumerate(tags_cat_file):
            spl = line.split()
            category = 'NULL'
            if len(spl) > CAT_COL:
                category = line.split()[CAT_COL]
                
            abbrv = CATEG_ABBRV[category]
            categ_id = CAT_IDS[abbrv]
            
            data.append(1)
            row.append(i)
            col.append(categ_id)
            
            new_col_ids[categ_id] = 'CAT_%s' % abbrv
        
        X_categ = np.asarray(sparse.coo_matrix((data, (row, col))).todense())
        return hstack_if_possible(X, X_categ), \
            update_col_ids(new_col_ids, column_ids)
            
def create_input_table(referrers_fpath = None, tseries_fpath = None, 
                       tags_cat_fpath = None, num_pts = 3):
    
    X = None
    column_ids = None
    
    if referrers_fpath:
        X, column_ids = load_referrers(referrers_fpath)
        
    if tseries_fpath and num_pts > 0:
        X, column_ids = load_time_series(tseries_fpath, num_pts, X, column_ids)
    
    if tags_cat_fpath:
        X, column_ids = load_categories(tags_cat_fpath, X, column_ids)
    
    inverse_names = dict((v, k) for k, v in column_ids.items())
    return X, column_ids, inverse_names

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
