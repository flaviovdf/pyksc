# -*- coding: utf8

from __future__ import division, print_function

from pyksc import regression

from scripts.class_predict import create_input_table
from scripts.class_predict import create_grid_search_cv

from sklearn import ensemble
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

import plac
import numpy as np
import sys

@plac.annotations(features_fpath=plac.Annotation('Input file', type=str),
                  tseries_fpath=plac.Annotation('Time series file', type=str),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str))
def main(features_fpath, tseries_fpath, assign_fpath):
    X = create_input_table(features_fpath, tseries_fpath)
    
    y_regr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)
    y_clf = np.genfromtxt(assign_fpath)

    clf = create_grid_search_cv('extra_trees', sparse = False, n_jobs=-1)
    
    regr = ensemble.ExtraTreesRegressor()
    mclass_rse_lstsq = regression.MultiClassRegression(clf, 
                                                       regr, n_jobs=-1)
    
    mrse = regression.mean_relative_square_error
    
    for test_size in [0.05, 0.25, 0.5]:
        cv = StratifiedShuffleSplit(y_clf, test_size=test_size, indices=False)
        
        f1_scores = []
        all_mrse = []
        all_r2 = []
        for train, test in cv:
            model = mclass_rse_lstsq.fit(X[train], y_clf[train], y_regr[train])
            X_test = X[test]
            
            y_clf_true = y_clf[test]
            y_rgr_true = y_regr[test]
            y_clf_pred, y_rgr_pred = model.predict(X_test, True)
            
            f1_scores.append(f1_score(y_clf_true, y_clf_pred))
            all_mrse.append(mrse(y_rgr_true, y_rgr_pred))
            all_r2.append(r2_score(y_rgr_true, y_rgr_pred))

        print('F1 - mean: ', np.mean(f1_scores))
        print('R2 - mean: ', np.mean(all_r2))
        
if __name__ == '__main__':
    sys.exit(plac.call(main))
