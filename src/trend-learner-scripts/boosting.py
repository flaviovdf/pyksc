#!/usr/bin/env python
from __future__ import division, print_function

from sklearn.base import clone

import numpy as np

EPS = 1e6

def sample_with_rep(weights, num_samples):

    weights = np.asanyarray(weights)

    assert weights.sum() >= (1 - EPS) and weights.sum() <= (1 + EPS)

    x = np.arange(weights.shape[0])
    y = np.random.multinomial(num_samples, weights)
    return np.repeat(x, y)

def error(y_true, y_pred, weights = None):

    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)

    if weights is not None:
        weights = np.asanyarray(weights)
    else:
        weights = np.ones(y_true.shape)

    data = np.asanyarray(y_true != y_pred, dtype='i')
    return (data * weights).sum() / weights.sum()

def comp_alpha(err, num_classes):

    return np.log((1 - err) / err) + np.log(num_classes - 1)

def compute_weights(y_true, y_pred, old_weights, alpha):
    
    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)
    old_weights = np.asanyarray(old_weights)

    return old_weights * np.exp(alpha * (y_true != y_pred))

def get_alpha_and_weights(y_true, y_pred, old_weights):

    num_classes = len(set(y_true))
    alpha = comp_alpha(error(y_true, y_pred), num_classes)
    weights = compute_weights(y_true, y_pred, old_weights, alpha)
    
    weights /= weights.sum()
    return alpha, weights

class ClassBoost(object):

    def __init__(self, classifier, sample_factor = 2.5):
        self.classifier = clone(classifier)
        self.sample_factor = sample_factor
        self.base_w = 0
        self.class_w = 0

    def fit(self, X, y, B):
        
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        B = np.asanyarray(B)

        ypred_base = np.asanyarray(B).argmax(axis = 1)

        assert X.shape[0] == y.shape[0] == ypred_base.shape[0]

        n = X.shape[0]

        uni_weights = np.ones(n) / n
        base_alpha, base_weights = get_alpha_and_weights(y, ypred_base, 
                uni_weights)
        
        #Sampling with repetition
        num_samples = int(n * self.sample_factor)
        idx = sample_with_rep(base_weights, num_samples)

        #Fitting
        X_new = X[idx]
        y_new = y[idx]
        self.classifier.fit(X_new, y_new)
        y_pred_new = self.classifier.predict(X)

        class_alpha, class_weights = get_alpha_and_weights(y, y_pred_new,
                base_weights)

        self.base_w = base_alpha
        self.class_w = class_alpha

    def predict(self, X, B):
        P_class = self.classifier.predict_proba(X)
        return (B * self.base_w + P_class * self.base_w).argmax(axis = 1)
