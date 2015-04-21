# -*- coding: utf8

from __future__ import division, print_function

from pyksc.trend import TrendLearner

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

import argparse
import ioutil
import numpy as np
import os
import plac
import sys

def fit(Xtrain, y_train, Xtest, num_pts):

    learner = TrendLearner(num_pts, 1)
    learner.fit(Xtrain, y_train)
    probs  = learner.predict_proba(Xtest)
        
    return probs

def main(tseries_fpath, centroids_fpath, test_fpath, assign_fpath, out_folder):
    
    C = np.genfromtxt(centroids_fpath)
    Xtest = ioutil.load_series(tseries_fpath, test_fpath)
    y_train = np.arange(C.shape[0])

    max_pts = Xtest.shape[1]
    for num_pts in range(1, max_pts + 1):
    #for num_pts in [1, 25, 50, 75]:
        probs = fit(C, y_train, Xtest, num_pts)

        probs_fpath = os.path.join(out_folder, 'probs-%d-pts.dat' % num_pts)
        np.savetxt(probs_fpath, probs)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))
