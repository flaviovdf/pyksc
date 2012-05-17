# -*- coding: utf8

from __future__ import division, print_function

from pyksc import regression

from scripts.learn_base import create_input_table
from scripts.learn_base import create_grid_search
from scripts.learn_base import clf_summary

from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score

from vod.stats.ci import half_confidence_interval_size as hci

import plac
import numpy as np
import sys

def create_learners(learner_name='extra_trees'):
    clf = create_grid_search(learner_name, n_jobs=-1)
    rgr = create_grid_search(learner_name, regressor=True, n_jobs=-1)

    return clf, rgr
    
def fit_and_predict(learner, rgr_base, X, y_clf, y_regr, train, test):
    model = learner.fit(X[train], y_clf[train])
        
    y_clf_true = y_clf[test]
    y_rgr_true = y_regr[test]
    y_clf_pred = model.predict(X[test])
    
    class_scores = np.array(precision_recall_fscore_support(y_clf_true,
                                                            y_clf_pred))
    
    micro_f1 = f1_score(y_clf_true, y_clf_pred, average='micro')
    macro_f1 = f1_score(y_clf_true, y_clf_pred, average='macro')
    
    y_rgr_base = rgr_base.fit(X[train], y_regr[train]).predict(X[test])
    general_r2 = r2_score(y_rgr_true, y_rgr_base)
    
    best_feat_clf = model.best_estimator_.feature_importances_
    best_feat_rgr = rgr_base.best_estimator_.feature_importances_
        
    return class_scores, micro_f1, macro_f1, general_r2, best_feat_clf, \
            best_feat_rgr

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

def print_results(clf_scores, micro, macro, r2_all):
    metric_means = np.mean(clf_scores, axis=0)
    metric_ci = hci(clf_scores, .95, axis=0)
    
    print(clf_summary(metric_means, metric_ci))
    print()
    print('Micro F1 - mean: %f +- %f' % (np.mean(micro), hci(micro, .95)))
    print('Macro F1 - mean: %f +- %f' % (np.mean(macro), hci(macro, .95)))
    print('R2 all   - mean: %f +- %f' % (np.mean(r2_all), hci(r2_all, .95)))


def print_final_summary(feature_ids, clf_scores, micro, macro, 
                        r2_all, importance_clf, importance_rgr):
    
    print_results(clf_scores, micro, macro, r2_all)
    print_importance(feature_ids, importance_clf, importance_rgr)

def run_experiment(X, y_clf, y_regr, feature_ids):
    
    clf_scores = []
    micro = []
    macro = []
    r2_all = []
    importance_clf = []
    importance_rgr = []
    
    learner, rgr_base = create_learners()
    cv = StratifiedKFold(y_clf, k=5)
    for train, test in cv:
        class_scores, micro_f1, macro_f1, general_r2, best_feat_clf, \
                best_feat_rgr = \
                fit_and_predict(learner, rgr_base, X, y_clf, y_regr, train, test)
        
        clf_scores.append(class_scores)
        micro.append(micro_f1)
        macro.append(macro_f1)
        
        r2_all.append(general_r2)
        importance_clf.append(best_feat_clf)
        importance_rgr.append(best_feat_rgr)
        
    print_final_summary(feature_ids, clf_scores, micro, macro, r2_all, 
                        importance_clf, importance_rgr)

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
    
    L, _, all_feat_names = create_input_table(all_features_fpath)
    X, feature_ids, feature_names = \
            create_input_table(partial_features_fpath, tseries_fpath, 
                               tag_categ_fpath, num_pts = num_days_to_use)
    
    #Sanity check
    for col_name in feature_names:
        if col_name in all_feat_names:
            assert feature_names[col_name] == all_feat_names[col_name]
            
    #Sort X by upload date
    #up_date_col = feature_names['A_UPLOAD_DATE']
    #sort_by_date = X[:,up_date_col].argsort()
    #X = X[sort_by_date]
    
    y_clf = np.genfromtxt(assign_fpath)
    y_regr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)
    
    print('Using all time features')
    run_experiment(X, y_clf, y_regr, feature_ids)

if __name__ == '__main__':
    sys.exit(plac.call(main))
