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
from sklearn.preprocessing import scale

from vod.stats.ci import half_confidence_interval_size as hci

import plac
import numpy as np
import numpy.ma as ma
import sys

def create_learners():
    clf = create_grid_search('extra_trees', n_jobs=-1)
    rgr = create_grid_search('extra_trees', regressor=True, n_jobs=-1)

    rgr_base = clone(rgr)
    learner = regression.MultiClassRegression(clf, rgr)

    return learner, rgr_base
    
def fit_and_predict(learner, rgr_base, X, y_clf, y_regr, train, test):
    model = learner.fit(X[train], y_clf[train], y_regr[train])
        
    y_clf_true = y_clf[test]
    y_rgr_true = y_regr[test]
    y_clf_pred, y_rgr_pred = model.predict(X[test], True)
    
    class_scores = np.array(precision_recall_fscore_support(y_clf_true, 
                                                            y_clf_pred))
    
    micro_f1 = f1_score(y_clf_true, y_clf_pred, average='micro')
    macro_f1 = f1_score(y_clf_true, y_clf_pred, average='macro')
    
    y_rgr_base = rgr_base.fit(X[train], y_regr[train]).predict(X[test])
    
    per_class_r2 = r2_score(y_rgr_true, y_rgr_pred)
    general_r2 = r2_score(y_rgr_true, y_rgr_base)
    
    best_feat_clf = model.clf_model.best_estimator_.feature_importances_
    best_feat_rgr = rgr_base.best_estimator_.feature_importances_
        
    return class_scores, micro_f1, macro_f1, per_class_r2, general_r2, \
            best_feat_clf, best_feat_rgr


def replace_missing_features(L, X, y, feature_ids, feature_names):
    
    #TODO: This id structure is kind of a hack
    ref_date_feats = []
    ref_view_feats = []
    point_feats = []
    for name, feat_id in feature_names.iteritems():
        if 'X_' in name:
            ref_date_feats.append(feat_id)

        if 'Y_' in name:
            ref_view_feats.append(feat_id)

        if 'POINT' in name:
            point_feats.append(feat_id)

    ref_date_feats.sort()
    ref_view_feats.sort()
    point_feats.sort()

    #Referrer View Features
    mean_feat_views = (L.T / y).T[:,ref_view_feats].mean(axis=0)
    views_so_far = X[:,point_feats].sum(axis=1)
    
    for row in xrange(X.shape[0]):
        for i, date_col in enumerate(ref_date_feats):
            view_col = ref_view_feats[i]
            assert feature_ids[view_col].split('_')[1] == \
                     feature_ids[date_col].split('_')[1]
                     
            if X[row, date_col] != 0:
                X[row, view_col] = views_so_far[row] * mean_feat_views[i]

    return X

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

def print_results(clf_scores, micro, macro, r2_class, r2_all):
    metric_means = np.mean(clf_scores, axis=0)
    metric_ci = hci(clf_scores, .95, axis=0)
    
    print(clf_summary(metric_means, metric_ci))
    print()
    print('Micro F1 - mean: %f +- %f' % (np.mean(micro), hci(micro, .95)))
    print('Macro F1 - mean: %f +- %f' % (np.mean(macro), hci(macro, .95)))
    print('R2 class - mean: %f +- %f' % (np.mean(r2_class), hci(r2_class, .95)))
    print('R2 all   - mean: %f +- %f' % (np.mean(r2_all), hci(r2_all, .95)))

def run_experiment(X, y_clf, y_regr, test_size, feature_ids):
    
    clf_scores = []
    micro = []
    macro = []
    r2_class = []
    r2_all = []

    importance_clf = []
    importance_rgr = []
    
    learner, rgr_base = create_learners()
    cv = StratifiedShuffleSplit(y_clf, test_size=test_size, indices=False)
    for train, test in cv:
        class_scores, micro_f1, macro_f1, per_class_r2, general_r2, \
        best_feat_clf, best_feat_rgr = fit_and_predict(learner, rgr_base, X, 
                                                       y_clf, y_regr, train, 
                                                       test)
        
        clf_scores.append(class_scores)
        micro.append(micro_f1)
        macro.append(macro_f1)
        
        r2_class.append(per_class_r2)
        r2_all.append(general_r2)

        importance_clf.append(best_feat_clf)
        importance_rgr.append(best_feat_rgr)

    print('Test Size = ', test_size)
    print_results(clf_scores, micro, macro, r2_class, r2_all)
    print()
    print_importance(feature_ids, importance_clf, importance_rgr)
    print('----')
        
@plac.annotations(all_features_fpath=plac.Annotation('All time features', 
                                                     type=str),
                  partial_features_fpath=plac.Annotation('Partial Features', 
                                                         type=str),
                  tag_categ_fpath=plac.Annotation('Tags file', type=str),
                  tseries_fpath=plac.Annotation('Time series file', type=str),
                  num_days_to_use=plac.Annotation('Num Days Series', type=int),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str))
def main(all_features_fpath, partial_features_fpath, tag_categ_fpath, 
         tseries_fpath, num_days_to_use, assign_fpath):
    
    A, _, all_feat_names = create_input_table(all_features_fpath)
    X, partial_feat_ids, partial_feat_names = \
            create_input_table(partial_features_fpath, tseries_fpath, 
                               tag_categ_fpath, num_pts = num_days_to_use)
    
    print(X.shape)
    print(partial_feat_ids)
    
    #Sanity check
    for col_name in partial_feat_names:
        if col_name in all_feat_names:
            assert partial_feat_names[col_name] == all_feat_names[col_name]
    
    y_clf = np.genfromtxt(assign_fpath)
    y_regr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)

    X = replace_missing_features(A, X, y_regr, partial_feat_ids, 
		                         partial_feat_names).copy()
    
    for test_size in [0.05, 0.1, 0.15, 0.20, 0.25]:
        run_experiment(X, y_clf, y_regr, test_size, partial_feat_ids)
        
if __name__ == '__main__':
    sys.exit(plac.call(main))
