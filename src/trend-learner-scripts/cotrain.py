#!/usr/bin/env python
from __future__ import division, print_function

from scipy import stats

from sklearn.base import clone

import numpy as np

class CoTrain(object):

    def __init__(self, classifier, label_fract = .25):
        self.classifier = clone(classifier)
        self.label_fract = label_fract

    def fit(self, X, y, P):
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        P = np.asanyarray(P)

        assert X.shape[0] == y.shape[0] == P.shape[0]

        n = X.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)

        n_classes = len(set(y))
        n_labelled = int(n * self.label_fract)

        init_labelled = idx[:n_labelled]
        w_label = np.zeros(n, dtype='bool')
        w_label[init_labelled] = True

        classes = np.arange(n_classes)
        y_new = np.zeros(n) - 1
        y_new[init_labelled] = y[init_labelled]
        while not w_label.all():
            self.classifier.fit(X[w_label], y_new[w_label])
            P_cls = self.classifier.predict_proba(X[~w_label])

            best_c = P_cls.argmax(axis = 0)
            best_p = P[~w_label].argmax(axis = 0)
            
            idx_c = np.where(~w_label)[0][best_c]
            idx_p = np.where(~w_label)[0][best_p]

            w_label[idx_c] = True
            w_label[idx_p] = True

            y_new[idx_c] = classes
            y_new[idx_p] = classes

    def predict(self, X, P):
        P_class = self.classifier.predict_proba(X)
        return (P * P_class).argmax(axis = 1)
