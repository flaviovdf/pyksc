# -*- coding: utf8

from __future__ import division, print_function

from pyksc import regression

from scripts.learn_base import create_input_table
from scripts.learn_base import create_grid_search
from scripts.learn_base import clf_summary

from sklearn.base import clone
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score

from vod.stats.ci import half_confidence_interval_size as hci

import plac
import numpy as np
import sys

@plac.annotations(features_fpath=plac.Annotation('Input file', type=str),
                  tseries_fpath=plac.Annotation('Time series file', type=str),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str))
def main(features_fpath, tseries_fpath, assign_fpath):
    
    X, col_names = create_input_table(features_fpath, None)
    y_regr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)
    y_clf = np.genfromtxt(assign_fpath)

    clf = create_grid_search('extra_trees', n_jobs=-1)
    rgr = create_grid_search('extra_trees', regressor=True, n_jobs=-1)
    learner = regression.MultiClassRegression(clf, rgr)
    
    rgr_base = clone(rgr)
    
    for test_size in [0.05, 0.25, 0.5]:
        cv = StratifiedShuffleSplit(y_clf, test_size=test_size, indices=False)
        
        clf_scores = []
        micro = []
        macro = []
        r2_class = []
        r2_all = []

        importance_clf = []
        importance_rgr = []

        for train, test in cv:
            model = learner.fit(X[train], y_clf[train], y_regr[train])
            
            y_clf_true = y_clf[test]
            y_rgr_true = y_regr[test]
            y_clf_pred, y_rgr_pred = model.predict(X[test], True)
            
            scores = np.array(precision_recall_fscore_support(y_clf_true, 
                                                              y_clf_pred))
            clf_scores.append(scores)
            micro.append(f1_score(y_clf_true, y_clf_pred, average='micro'))
            macro.append(f1_score(y_clf_true, y_clf_pred, average='macro'))
            
            r2_class.append(r2_score(y_rgr_true, y_rgr_pred))

            y_rgr_base = rgr_base.fit(X[train], y_regr[train]).predict(X[test])
            r2_all.append(r2_score(y_rgr_true, y_rgr_base))

            best_feat_clf = model.clf_model.best_estimator_.feature_importances_
            best_feat_rgr = rgr_base.best_estimator_.feature_importances_
            
            importance_clf.append(best_feat_clf)
            importance_rgr.append(best_feat_rgr)

        metric_means = np.mean(clf_scores, axis=0)
        metric_ci = hci(clf_scores, .95, axis=0)
        
        print("Test Size = ", test_size)
        print(clf_summary(metric_means, metric_ci))
        print()
        print('Macro F1 - mean: ', np.mean(micro))
        print('Micro F1 - mean: ', np.mean(macro))
        print('R2 class - mean: ', np.mean(r2_class))
        print('R2 all   - mean: ', np.mean(r2_all))
        print()
        
        clf_imp = np.mean(importance_clf, axis=0)
        rgr_imp = np.mean(importance_rgr, axis=0)
        print('Classif Importance')
        for key in clf_imp.argsort()[::-1]:
            print(col_names[key], clf_imp[key])
        
        print('Regr Importance')
        for key in rgr_imp.argsort()[::-1]:
            print(col_names[key], rgr_imp[key])
        
        print('---')
        
if __name__ == '__main__':
    sys.exit(plac.call(main))
