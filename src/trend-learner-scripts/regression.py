#!/usr/bin/env python
from __future__ import division, print_function

from collections import defaultdict

from vod.stats.ci import half_confidence_interval_size as hci

import numpy as np
import plac
import sys

class OLS(object):

    def __init__(self):
        
        self.coeffs = None
        self.residuals = None
        self.gcv_sqerrors = None

    def fit(self, X, y):

        assert X.shape[0] == y.shape[0]

        X = np.asanyarray(X, dtype='f', order='C')
        y = np.asanyarray(y, dtype='f', order='C')

        n = y.shape[0]

        PI = np.linalg.pinv(X)
        H = np.dot(X, PI)
        self.coeffs = np.dot(PI, y)
        
        y_hat = np.dot(X, self.coeffs)
        
        self.residuals = y_hat - y

        aux = self.residuals / (1 - np.diag(H))
        self.gcv_sqerrors = np.power(aux, 2)

    def predict(X):
        return np.dot(X, self.coeffs)

def fit(X, tr, tt):
    ols = OLS()
    
    y = X[:, :tt].sum(axis=1)
    XR = (X[:, :tr].T / y).T
    ols.fit(XR, np.ones(XR.shape[0]))

    return ols

def main(tseries_fpath, predict_fpath, bestby_fpath):

    X = np.genfromtxt(tseries_fpath)[:,1:] + 0.0001
    cls_pred = np.loadtxt(predict_fpath, dtype='i')
    rgr_true = X.sum(axis=1)
    bestby = np.genfromtxt(bestby_fpath)

    cls_labels = set(cls_pred[cls_pred != -1])

    tt = X.shape[1]
    models = {}
    models_per_clust = {}
    ref_time = np.arange(1, tt + 1)

    #tr = 7
    #ref_time = np.array([tr])
    #bestby = np.zeros(bestby.shape[0]) + tr

    for tr in ref_time:
        models[tr] = fit(X, tr, tt)
        
        for k in sorted(cls_labels):
            Xk = X[cls_pred == k]
            models_per_clust[tr, k] = fit(Xk, tr, tt)
    
    errors_all = []
    errors_cls = []
    errors_per_cls = defaultdict(list)
    for tr in ref_time:
        idx = bestby == tr
        ols = models[tr]

        errors_all.extend(ols.gcv_sqerrors[idx])
        classes = cls_pred[idx]

        for cls in set(classes):
            bestby_for_cls = bestby[cls_pred == cls]
            idx_cls = bestby_for_cls == tr
            
            ols = models_per_clust[tr, cls]
            errors_cls.extend(ols.gcv_sqerrors[idx_cls])
            errors_per_cls[cls].extend(ols.gcv_sqerrors[idx_cls])

    print('Glob model:', np.mean(errors_all), '+-', hci(errors_all, .95))
    print('Spec model:', np.mean(errors_cls), '+-', hci(errors_cls, .95))
    print()
    print('Per class')
    for cls in cls_labels:
        err = errors_per_cls[cls]
        print('Cls = ', cls, np.mean(err), '+-', hci(err, .95))


if __name__ == '__main__':
    sys.exit(plac.call(main))
