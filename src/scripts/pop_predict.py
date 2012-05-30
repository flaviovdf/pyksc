# -*- coding: utf8

from __future__ import division, print_function

from scripts.learn_base import create_input_table
from scripts.learn_base import create_grid_search
from scripts.learn_base import clf_summary

from pyksc.regression import mean_relative_square_error as mrse

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale

from vod.stats.ci import half_confidence_interval_size as hci

import plac
import numpy as np
import os
import sys

def create_learners(learner_name='rbf_svm'):
    clf = create_grid_search(learner_name, n_jobs=-1)
    rgr = create_grid_search(learner_name, regressor=True, n_jobs=-1)

    return clf, rgr
    
def fit_and_predict(clf, rgr, X, y_clf, y_rgr, train, test, out_folder, fold):
    clf_model = clf.fit(X[train], y_clf[train])
        
    y_clf_true = y_clf[test]
    y_rgr_true = y_rgr[test]
    y_clf_pred = clf_model.predict(X[test])
    
    class_scores = np.array(precision_recall_fscore_support(y_clf_true,
                                                            y_clf_pred))
    micro_f1 = f1_score(y_clf_true, y_clf_pred, average='micro')
    macro_f1 = f1_score(y_clf_true, y_clf_pred, average='macro')
    
    rgr_model = rgr.fit(X[train], y_rgr[train])
    y_rgr_pred = rgr_model.predict(X[test])
    
    general_r2 = r2_score(y_rgr_true, y_rgr_pred)
    mse_score  = mse(y_rgr_true, y_rgr_pred)
    mrse_score = mrse(y_rgr_true, y_rgr_pred)
    
    clf_pred_fpath = os.path.join(out_folder, '%d-clf.pred' % fold)
    clf_true_fpath = os.path.join(out_folder, '%d-clf.true' % fold)
    
    rgr_pred_fpath = os.path.join(out_folder, '%d-rgr.pred' % fold)
    rgr_true_fpath = os.path.join(out_folder, '%d-rgr.true' % fold)
    
    np.savetxt(clf_pred_fpath, y_clf_pred, fmt="%d")
    np.savetxt(clf_true_fpath, y_clf_true, fmt="%d")
    
    np.savetxt(rgr_pred_fpath, y_rgr_pred)
    np.savetxt(rgr_true_fpath, y_rgr_true)
    
    return class_scores, micro_f1, macro_f1, general_r2, mse_score,\
            mrse_score

def print_results(clf_scores, micro, macro, r2_all, mse_all, mrse_all):
    metric_means = np.mean(clf_scores, axis=0)
    metric_ci = hci(clf_scores, .95, axis=0)
    
    print(clf_summary(metric_means, metric_ci))
    print('Micro F1 - mean: %f +- %f' % (np.mean(micro), hci(micro, .95)))
    print('Macro F1 - mean: %f +- %f' % (np.mean(macro), hci(macro, .95)))
    print('R2 all   - mean: %f +- %f' % (np.mean(r2_all), hci(r2_all, .95)))
    print('MSE all   - mean: %f +- %f' % (np.mean(mse_all), hci(mse_all, .95)))
    print('MRSE all   - mean: %f +- %f' % (np.mean(mrse_all), 
                                           hci(mrse_all, .95)))

def run_experiment(X, y_clf, y_regr, feature_ids, out_folder):
    
    clf_scores = []
    micro = []
    macro = []
    r2_all = []
    mse_all = []
    mrse_all = []
    
    learner, rgr_base = create_learners()
    cv = StratifiedKFold(y_clf, k=5)
    fold_num = 1
    for train, test in cv:
        class_scores, micro_f1, macro_f1, general_r2, \
                mse_score, mrse_score = \
                fit_and_predict(learner, rgr_base, X, y_clf, y_regr, train, 
                                test, out_folder, fold_num)
        
        clf_scores.append(class_scores)
        micro.append(micro_f1)
        macro.append(macro_f1)
        
        r2_all.append(general_r2)
        mse_all.append(mse_score)
        mrse_all.append(mrse_score)

        fold_num += 1
        
    print_results(clf_scores, micro, macro, r2_all, mse_all, mrse_all)

@plac.annotations(features_fpath=plac.Annotation('Partial Features', 
                                                         type=str),
                  tag_categ_fpath=plac.Annotation('Tags file', type=str),
                  tseries_fpath=plac.Annotation('Time series file', type=str),
                  num_days_to_use=plac.Annotation('Num Days Series', type=int),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str),
                  out_foldpath=plac.Annotation('Output folder', type=str))
def main(features_fpath, tag_categ_fpath, tseries_fpath, num_days_to_use, 
         assign_fpath, out_foldpath):
    
    X, feature_ids, _ = \
            create_input_table(features_fpath, tseries_fpath, tag_categ_fpath,
                               num_days_to_use)
   
    X = scale(X)
    y_clf = np.genfromtxt(assign_fpath)
    y_regr = scale(np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1))
    run_experiment(X, y_clf, y_regr, feature_ids, out_foldpath)

if __name__ == '__main__':
    sys.exit(plac.call(main))
