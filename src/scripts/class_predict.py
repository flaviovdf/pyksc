# -*- coding: utf8

from __future__ import division, print_function

from scripts.learn_base import create_input_table
from scripts.learn_base import create_grid_search
from scripts.learn_base import clf_summary

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import scale

from vod.stats.ci import half_confidence_interval_size as hci

import argparse
import numpy as np
import sys
import traceback

def run_classifier(clf, X, y):
    n_folds = 5
    cross_fold = StratifiedKFold(y, k=n_folds)
    
    #class_matrices has shape [n_folds, 4, n_classes] 
    #The second dimension has 4 metrics: for precision, recall, f1, support
    R_cv = cross_val_score(clf, X, y, cv=cross_fold, n_jobs=1, 
                           score_func=precision_recall_fscore_support)
    
    C_cv = cross_val_score(clf, X, y, cv=cross_fold, n_jobs=1, 
                           score_func=confusion_matrix)

    class_matrices = []
    conf_matrices = []
    for i in xrange(n_folds):
        class_matrices.append(R_cv[i])
        
        conf_matrix_aux = 1.0 * C_cv[i]
        conf_matrix_aux = (conf_matrix_aux.T / conf_matrix_aux.sum(axis=1)).T
        conf_matrices.append(conf_matrix_aux)

    return class_matrices, conf_matrices
    
def main(features_fpath, tseries_fpath, tags_fpath, classes_fpath, clf_name):
    X, params = create_input_table(features_fpath, tseries_fpath, tags_fpath)
    y = np.loadtxt(classes_fpath)
    
    clf = create_grid_search(clf_name)
    class_matrices, conf_matrices = run_classifier(clf, X, y)
    
    metric_means = np.mean(class_matrices, axis=0)
    metric_ci = hci(class_matrices, .95, axis=0)
    print(clf_summary(metric_means, metric_ci))
    print()
    
    conf_means = np.mean(conf_matrices, axis=0)
    conf_ci = hci(conf_matrices, .95, axis=0)
    print("Average confusion matrix with .95 confidence interval")
    print(" \ttrue ")
    print("predic")
    for i in xrange(conf_means.shape[0]):
        print(i, end="\t \t")
        for j in xrange(conf_means.shape[1]):
            print('%.3f +- %.3f' % (conf_means[i, j], conf_ci[i, j]), end="\t")
        print()

def create_parser(prog_name):
    
    desc = __doc__
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog_name, description=desc, 
                                     formatter_class=formatter)
    
    parser.add_argument('--features_fpath', type=str,
                        help='Input file with video features')
    parser.add_argument('--tseries_fpath', type=str,
                        help='Input file with video time series')
    parser.add_argument('--tags_fpath', type=str,
                        help='Input file with video tags')
    parser.add_argument('classes_fpath', type=str,
                        help='Classes to predict')
    parser.add_argument('clf_name', type=str, choices=['rbf_svm', 
                                                       'linear_svm',
                                                       'extra_trees'],
                        help='Classifier to use')
    
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.features_fpath, values.tseries_fpath, 
                    values.tags_fpath, values.classes_fpath, values.clf_name)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))
