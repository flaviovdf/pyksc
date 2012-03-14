# -*- coding: utf8

from __future__ import division, print_function

from scipy.sparse import hstack

from scripts.tags_io import vectorize_videos

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale

from sklearn import ensemble
from sklearn import svm

from vod.stats.ci import t_table

import argparse
import numpy as np
import sys
import traceback

def get_classifier_and_params(name, sparse = False):
    
    clf = None
    param_grid = None
    
    if name == 'rbf_svm':
        param_grid = {
                      'C':[0.1, 0.5, 1, 5, 10, 50, 100], 
                      'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                      'scale_C':[False]
                      }
        
        if sparse:
            clf = svm.sparse.SVC(kernel='rbf', cache_size=2048)
        else:
            clf = svm.SVC(kernel='rbf', cache_size=2048)
    elif name == 'linear_svm':
        param_grid = {
                      'C':[0.1, 0.5, 1, 5, 10, 50, 100],
                      'scale_C':[False]
                      }
        
        if sparse:
            clf = svm.sparse.LinearSVC()
        else:
            clf = svm.LinearSVC()
    elif name == 'extra_trees':
        param_grid = {
                      'n_estimators':[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                      'min_split':[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                      }
        clf = ensemble.ExtraTreesClassifier()
    else:
        raise Exception('Unknown Classifier')
    
    return clf, param_grid

def create_grid_search_cv(name, sparse):
    clf, params = get_classifier_and_params(name, sparse)
    grid_search = GridSearchCV(clf, params, score_func=f1_score, 
                               cv=3, refit=True)
    return grid_search

def run_classifier(clf, X, y):
    n_folds = 10
    cross_fold = StratifiedKFold(y, k=n_folds)
    
    #class_matrices has shape [n_folds, 4, n_classes] 
    #The second dimension has 4 metrics: for precision, recall, f1, support
    R_cv = cross_val_score(clf, X, y, cv=cross_fold, n_jobs=-1, 
                           score_func=precision_recall_fscore_support)
    
    C_cv = cross_val_score(clf, X, y, cv=cross_fold, n_jobs=-1, 
                           score_func=confusion_matrix)

    class_matrices = []
    conf_matrices = []
    for i in xrange(n_folds):
        class_matrices.append(R_cv[i])
        
        conf_matrix_aux = 1.0 * C_cv[i]
        conf_matrix_aux = (conf_matrix_aux.T / conf_matrix_aux.sum(axis=1)).T
        conf_matrices.append(conf_matrix_aux)

    return class_matrices, conf_matrices
    
def get_mean_std_and_ci(matrices):
    
    n_matrices = len(matrices)
    
    means = sum(matrices) / n_matrices
    
    stds = np.zeros(means.shape)
    for i in xrange(n_matrices):
        stds += (matrices[i] - means) ** 2
    stds /= n_matrices - 1
    
    cis = t_table(n_matrices - 1, 0.95) * stds / np.sqrt(n_matrices)

    return means, stds, cis

def main(features_fpath, tags_fpath, classes_fpath, clf_name):
    if features_fpath is None and tags_fpath is None:
        raise Exception('Both the features_fpath and tags_fpath cannot be None')

    X = None
    sparse = False
    if features_fpath is not None:
        print('Using referrers')
        X_features = scale(np.genfromtxt(features_fpath)[:,1:].copy())
        X = X_features
        
    if tags_fpath is not None:
        print('Using tags')
        X_tags = vectorize_videos(tags_fpath)[0]
        X = X_tags
        sparse = True
        
    if features_fpath is not None and tags_fpath is not None:
        print('Combining both')
        X = hstack([X_features, X_tags], format='csr')
        sparse = True
    
    print('Input shape', X.shape)
    y = np.loadtxt(classes_fpath)
    
    clf = create_grid_search_cv(clf_name, sparse)
    class_matrices, conf_matrices = run_classifier(clf, X, y)
    
    metric_means, metrics_std, metrics_ci = get_mean_std_and_ci(class_matrices)
    conf_means, conf_std, conf_ci = get_mean_std_and_ci(conf_matrices)
    
    print("Average metrics per class with .95 confidence intervals")
    print("class \tprecision \trecall \tf1 score \tsupport")
    for j in xrange(metric_means.shape[1]):
        print(j, end="\t")
        for i in xrange(metric_means.shape[0]):
            print('%.3f +- %.3f' % (metric_means[i, j], metrics_ci[i, j]), 
                  end="\t")
        print()
    print("--")
    print()
    
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
        return main(values.features_fpath, values.tags_fpath, 
                    values.classes_fpath, values.clf_name)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))
