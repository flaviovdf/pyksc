# -*- coding: utf8

from __future__ import division, print_function

from scipy.sparse import isspmatrix
from scipy.sparse import hstack

from scripts.tags_io import vectorize_videos

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale

from sklearn import svm

from vod.stats.ci import t_table

import argparse
import numpy as np
import sys
import traceback

def find_best_parameters(X_model, y_model, kernel):
    
    svc = None
    param_grid = None
    if kernel == 'rbf':
        print('Using rbf kernel')
        param_grid = {
                      'C':[0.1, 0.5, 1, 5, 10, 50, 100], 
                      'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
                      }
        
        if isspmatrix(X_model):
            svc = svm.sparse.SVC(kernel='rbf', cache_size=2048)
        else:
            svc = svm.SVC(kernel='rbf', cache_size=2048)
    else:
        print('Using linear kernel')
        param_grid = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
        
        if isspmatrix(X_model):
            svc = svm.sparse.LinearSVC()
        else:
            svc = svm.LinearSVC()
    
    clf = GridSearchCV(svc, param_grid, n_jobs=-1, score_func=f1_score)
    clf = clf.fit(X_model, y_model)
    best_clf = clf.best_estimator
    
    return best_clf

def run_classifier(X_model, y_model, X_cross, y_cross, kernel):
    best_clf = find_best_parameters(X_model, y_model, kernel)

    n_folds = 10
    cross_fold = StratifiedKFold(y_cross, k=n_folds)
    
    #class_metrics has shape [n_folds, 4, n_classes] 
    #The second dimension has 4 metrics: for precision, recall, f1, support
    R_cv = cross_val_score(best_clf, X_cross, y_cross, cv=cross_fold, n_jobs=-1, 
                           score_func=precision_recall_fscore_support)
    
    C_cv = cross_val_score(best_clf, X_cross, y_cross, cv=cross_fold, n_jobs=-1, 
                           score_func=confusion_matrix)

    class_metrics = np.zeros(shape=(R_cv.shape[1], R_cv.shape[2]), dtype='f')
    conf_matrix = np.zeros(shape=(C_cv.shape[1], C_cv.shape[2]), dtype='f')
    class_dists = np.zeros(shape=R_cv.shape[2], dtype='f')
    for i in xrange(n_folds):
        class_metrics += R_cv[i]
        
        conf_matrix_aux = 1.0 * C_cv[i]
        class_dists += conf_matrix_aux.sum(axis=1)
        
        conf_matrix_aux = (conf_matrix_aux.T / conf_matrix_aux.sum(axis=1)).T
        conf_matrix += conf_matrix_aux

    class_metrics /= n_folds
    conf_matrix /= n_folds
    class_dists /= n_folds

    return class_metrics, conf_matrix, class_dists
    
def get_mean_std_and_ci(matrices):
    
    n_matrices = len(matrices)
    
    means = sum(matrices) / n_matrices
    
    stds = np.zeros(means.shape)
    for i in xrange(n_matrices):
        stds += (matrices[i] - means) ** 2
    stds /= n_matrices - 1
    
    cis = t_table(n_matrices - 1, 0.95) * stds / np.sqrt(n_matrices)

    return means, stds, cis

def main(features_fpath, tags_fpath, classes_fpath, kernel):
    if features_fpath is None and tags_fpath is None:
        raise Exception('Both the features_fpath and tags_fpath cannot be None')

    X = None
    if features_fpath is not None:
        print('Using referrers')
        X_features = scale(np.genfromtxt(features_fpath)[:,1:].copy())
        X = X_features
        
    if tags_fpath is not None:
        print('Using tags')
        X_tags = vectorize_videos(tags_fpath)[0]
        X = X_tags
        
    if features_fpath is not None and tags_fpath is not None:
        print('Combining both')
        X = hstack([X_features, X_tags], format='csr')
    
    print('Input shape', X.shape)
    y = np.loadtxt(classes_fpath)
    
    n_features = X.shape[0]
    indexes = np.arange(X.shape[0])
    
    class_metrics_matrices  = []
    confusion_matrices = []
    class_dist_matrices = []
    
    n_runs = 10
    for i in xrange(n_runs):
        np.random.shuffle(indexes)
        X_rand = X[indexes]
        y_rand = y[indexes]
        
        X_model = X_rand[:n_features // 2]
        y_model = y_rand[:n_features // 2]
    
        X_cross = X_rand[n_features // 2:]
        y_cross = y_rand[n_features // 2:]
        
        class_metrics, conf_matrix, class_dists = \
            run_classifier(X_model, y_model, X_cross, y_cross, kernel)
            
        class_metrics_matrices.append(class_metrics)
        confusion_matrices.append(conf_matrix)
        class_dist_matrices.append(class_dists)
    
    metrics_means, metrics_std, metrics_ci = \
        get_mean_std_and_ci(class_metrics_matrices)
        
    conf_matrix_means, conf_matrix_std, conf_matrix_ci = \
        get_mean_std_and_ci(confusion_matrices)

    class_matrix_means, class_matrix_std, class_matrix_ci = \
        get_mean_std_and_ci(class_dist_matrices)            

    print("Average metrics per class with .95 confidence intervals")
    print("class \tprecision \trecall \tf1 score \tsupport")
    for j in xrange(metrics_means.shape[1]):
        print(j, end="\t")
        for i in xrange(metrics_means.shape[0]):
            print('%.3f +- %.3f' % (metrics_means[i, j], metrics_ci[i, j]), 
                  end="\t")
        print()
    print("--")
    print()
    
    print("Average confusion matrix with .95 confidence interval")
    labels = " \t".join(str(c) for c in xrange(conf_matrix_means.shape[0]))
    print(" \ttrue \t" + labels + " \tn_vids")
    print("predic")
    for i in xrange(conf_matrix_means.shape[0]):
        print(i, end="\t \t")
        for j in xrange(conf_matrix_means.shape[1]):
            print('%.3f +- %.3f' % (conf_matrix_means[i, j], 
                                    conf_matrix_ci[i, j]), end="\t")
        print('%.3f +- %.3f' % (class_matrix_means[i], class_matrix_ci[i]))            

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
    parser.add_argument('kernel', type=str, choices=['rbf', 'linear'],
                        help='Kernel to use [rbf or linear]')
    
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.features_fpath, values.tags_fpath, 
                    values.classes_fpath, values.kernel)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))