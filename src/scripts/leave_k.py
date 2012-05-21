# -*- coding: utf8

from __future__ import division, print_function

from scripts.learn_base import create_input_table
from scripts.learn_base import create_grid_search

from sklearn.metrics import zero_one_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_square_error as mse
from sklearn.metrics import r2_score

import plac
import numpy as np
import sys

def create_learners(learner_name='extra_trees'):
    clf = create_grid_search(learner_name, n_jobs=-1)
    rgr = create_grid_search(learner_name, regressor=True, n_jobs=-1)

    return clf, rgr
    
def print_importance(feature_ids, importance_clf, importance_rgr):
    clf_imp = np.mean(importance_clf, axis=0)
    rgr_imp = np.mean(importance_rgr, axis=0)
    
    print('Classification Importance')
    for key in clf_imp.argsort()[:-1]:
        print(feature_ids[key], clf_imp[key])
    
    print()
    print('Regression Importance')
    for key in rgr_imp.argsort()[:-1]:
        print(feature_ids[key], rgr_imp[key])

def mae(y_true, y_pred):
    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))

def run_experiment(X, y_clf, y_rgr, feature_ids, k=500):
    clf, rgr = create_learners()
    
    n = len(y_clf)
    train_index = np.ones(n, dtype=np.bool)
    train_index[-k:] = False
    test_index = np.logical_not(train_index)
    
    print(train_index.sum(), test_index.sum())
    
    clf_model = clf.fit(X[train_index], y_clf[train_index]) 
    rgr_model = rgr.fit(X[train_index], y_rgr[train_index])
    
    clf_true = y_clf[test_index]
    clf_predict = clf_model.predict(X[test_index])
    
    rgr_true = y_rgr[test_index]
    rgr_pred = rgr_model.predict(X[test_index])
    
    print('Micro F1: ', f1_score(clf_true, clf_predict, average='micro'))
    print('Macro F1: ', f1_score(clf_true, clf_predict, average='macro'))
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
                                               type=str))
def main(partial_features_fpath, tag_categ_fpath, tseries_fpath, 
         num_days_to_use, assign_fpath):
    
    X, feature_ids, feature_names = \
            create_input_table(partial_features_fpath, tseries_fpath, 
                               tag_categ_fpath, num_pts = num_days_to_use)
    
    #Sort X by upload date
    up_date_col = feature_names['A_UPLOAD_DATE']
    sort_by_date = X[:,up_date_col].argsort()
    X = X[sort_by_date].copy()
    
    y_clf = np.genfromtxt(assign_fpath)[sort_by_date]
    y_regr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)[sort_by_date]
    run_experiment(X, y_clf, y_regr, feature_ids)

if __name__ == '__main__':
    sys.exit(plac.call(main))