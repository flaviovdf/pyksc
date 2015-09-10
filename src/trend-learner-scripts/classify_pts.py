# -*- coding: utf8

from __future__ import division, print_function

from pyksc.trend import TrendLearner

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import ioutil
import numpy as np
import os
import plac
import sys

def fit(C, y_train, X, y_true, num_pts):

    learner = TrendLearner(num_pts, 1)
    learner.fit(C, y_train)

    probs  = learner.predict_proba(X)
    y_pred = probs.argmax(axis=1)
        
    return y_pred, probs

def main(tseries_fpath, train_fpath, centroids_fpath, classes_fpath, out_folder,
	 gamma_max):
    gamma_max = int(gamma_max)

    X = ioutil.load_series(tseries_fpath, train_fpath)
    C = np.genfromtxt(centroids_fpath, dtype='f')
    
    y_train = np.arange(C.shape[0])
    y_true = np.genfromtxt(classes_fpath)
    max_pts = gamma_max
    #max_pts = X.shape[1]

    best_by = np.zeros(X.shape[0])
    min_conf = np.zeros(X.shape[0])
    all_probs = np.zeros(shape=(X.shape[0], max_pts))

    lousy_conf = 1.0 / C.shape[0] #if confidence is equal to this, classifier did nothing
    for num_pts in range(1, max_pts + 1):
        y_pred, probs = fit(C, y_train, X, y_true, num_pts)

        for i in xrange(X.shape[0]):
            p_true = probs[i, y_true[i]]
            if best_by[i] == 0 and y_pred[i] == y_true[i] and p_true > lousy_conf:
                best_by[i] = num_pts
                min_conf[i] = probs[i, y_true[i]]
            all_probs[i, num_pts - 1] = p_true

        summary_fpath = os.path.join(out_folder,\
                'class_summ-%d-pts.dat' % num_pts)
        probs_fpath = os.path.join(out_folder, 'probs-%d-pts.dat' % num_pts)

        with open(summary_fpath, 'w') as summary_file:
            print(classification_report(y_true, y_pred), file=summary_file)
        np.savetxt(probs_fpath, probs)
    
    best_fpath = os.path.join(out_folder, 'best-by.dat')
    conf_fpath = os.path.join(out_folder, 'conf.dat')
    all_conf_fpath = os.path.join(out_folder, 'all-conf.dat')

    np.savetxt(best_fpath, best_by)
    np.savetxt(conf_fpath, min_conf)
    np.savetxt(all_conf_fpath, np.asarray(all_probs))

if __name__ == '__main__':
    sys.exit(plac.call(main))
