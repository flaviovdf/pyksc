# -*- coding: utf8

from __future__ import division, print_function

from scripts.learn_base import create_input_table
from scripts.learn_base import create_grid_search

from sklearn.metrics import f1_score
from sklearn.metrics import mean_square_error as mse
from sklearn.metrics import r2_score

import plac
import numpy as np
import os
import sys

def create_learners(learner_name='extra_trees'):
    clf = create_grid_search(learner_name, n_jobs=-1)
    rgr = create_grid_search(learner_name, regressor=True, n_jobs=-1)

    return clf, rgr
    
def print_importance(feature_ids, importance_clf, importance_rgr):
    print()    
    print('Classification Importance')
    for key in importance_clf.argsort()[::-1]:
        print(feature_ids[key], importance_clf[key])
    
    print()
    print('Regression Importance')
    for key in importance_rgr.argsort()[::-1]:
        print(feature_ids[key], importance_rgr[key])

def mae(y_true, y_pred):
    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))

def run_experiment(X, y_clf, y_rgr, feature_ids, out_foldpath, k=500):
    clf, rgr = create_learners()
    
    n = len(y_clf)
    train_index = np.ones(n, dtype=np.bool)
    train_index[-k:] = False
    test_index = np.logical_not(train_index)
    
    clf_model = clf.fit(X[train_index], y_clf[train_index]) 
    rgr_model = rgr.fit(X[train_index], y_rgr[train_index])
    
    clf_true = y_clf[test_index]
    clf_pred = clf_model.predict(X[test_index])
    
    rgr_true = y_rgr[test_index]
    rgr_pred = rgr_model.predict(X[test_index])
    
    clf_pred_fpath = os.path.join(out_foldpath, '%clf.pred')
    clf_true_fpath = os.path.join(out_foldpath, '%clf.true')
    
    rgr_pred_fpath = os.path.join(out_foldpath, '%rgr.pred')
    rgr_true_fpath = os.path.join(out_foldpath, '%rgr.true')
    
    np.savetxt(clf_pred_fpath, clf_pred, fmt="%d")
    np.savetxt(clf_true_fpath, clf_true, fmt="%d")
    
    np.savetxt(rgr_pred_fpath, rgr_pred)
    np.savetxt(rgr_true_fpath, rgr_true)
    
    print('Micro F1: ', f1_score(clf_true, clf_pred, average='micro'))
    print('Macro F1: ', f1_score(clf_true, clf_pred, average='macro'))
    print()
    print('R2: ', r2_score(rgr_true, rgr_pred))
    print('MAE: ', mae(rgr_true, rgr_pred))
    print('MSE: ', mse(rgr_true, rgr_pred))
    print()
    print_importance(feature_ids, 
                     clf_model.best_estimator_.feature_importances_,
                     rgr_model.best_estimator_.feature_importances_)

@plac.annotations(partial_features_fpath=plac.Annotation('Partial Features', 
                                                         type=str),
                  tag_categ_fpath=plac.Annotation('Tags file', type=str),
                  tseries_fpath=plac.Annotation('Time series file', type=str),
                  num_days_to_use=plac.Annotation('Num Days Series', type=int),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str),
                  out_foldpath=plac.Annotation('Output folder', type=str))
def main(partial_features_fpath, tag_categ_fpath, tseries_fpath, 
         num_days_to_use, assign_fpath, out_foldpath):
    
    X, feature_ids, feature_names = \
            create_input_table(partial_features_fpath, tseries_fpath, 
                               tag_categ_fpath, num_pts = num_days_to_use)
    
    #Sort X by upload date
    up_date_col = feature_names['A_UPLOAD_DATE']
    sort_by_date = X[:,up_date_col].argsort()
    X = X[sort_by_date].copy()
    
    y_clf = np.genfromtxt(assign_fpath)[sort_by_date]
    y_regr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)[sort_by_date]
    run_experiment(X, y_clf, y_regr, feature_ids, out_foldpath)

if __name__ == '__main__':
    sys.exit(plac.call(main))