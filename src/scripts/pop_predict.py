# -*- coding: utf8

from __future__ import division, print_function

from pyksc import regression

from scripts.class_predict import create_input_table
from scripts.class_predict import create_grid_search_cv

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score

import plac
import numpy as np
import sys

@plac.annotations(features_fpath=plac.Annotation('Input file', type=str),
                  tseries_fpath=plac.Annotation('Time series file', type=str),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str))
def main(features_fpath, tseries_fpath, assign_fpath):
    X = create_input_table(features_fpath, tseries_fpath).copy()
    
    y_regr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)
    y_clf = np.genfromtxt(assign_fpath)

    clf = create_grid_search_cv('rbf_svm', sparse = False, n_jobs=-1)
    regr = regression.RSELinearRegression()
    mclass_rse_lstsq = regression.MultiClassRegression(clf, 
                                                       regr, n_jobs=-1)
    
    mrse = regression.mean_relative_square_error
    for fract in [0.5, 0.75, 0.9]:
        cv = ShuffleSplit(n=X.shape[0], train_fraction=fract)
        iter = 0
        for train, test in cv:
            X_train = X[train]
            X_test = X[test]
            
            y_clf_train = y_clf[train]
            y_clf_test = y_clf[test]

            y_regr_train = y_regr[train]
            y_regr_test = y_regr[test]
            
            model = mclass_rse_lstsq.fit(X_train, y_clf_train, y_regr_train)
            y_clf_pred, y_regr_pred = model.predict(X_test, True)
            
            print('Classification Report: Train = %.2f; iteration %d ' \
                  %(fract, iter))
            print(classification_report(y_clf_test, y_clf_pred))
            print('-- Regression Analysis:')
            print('---- mRSE = %f' % mrse(y_regr_test, y_regr_pred))
            print('---- R2 = %f' % r2_score(y_regr_test, y_regr_pred))
            
            iter += 1
            
if __name__ == '__main__':
    sys.exit(plac.call(main))