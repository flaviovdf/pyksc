#-*- coding: utf8

import _trend

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

import numpy as np

class TrendLearner(BaseEstimator, ClassifierMixin):

    def __init__(self, gamma, num_steps):
        self.gamma = gamma
        self.num_steps = num_steps
        self.num_labels = 0

    def fit(self, X, y):
        self.R = np.asanyarray(X, dtype=np.float64, order='C')
        
        y = np.asanyarray(y)
        unique, labels_flat = np.unique(y, return_inverse=True)
        self.labels = labels_flat.reshape(y.shape)
        self.num_labels = unique.shape[0]
        

    def predict(self, X):

        X = np.asanyarray(X, dtype=np.float64, order='C')
        P = _trend.predict(X, self.R, self.labels, self.num_labels,
                           self.gamma, self.num_steps)
        
        return P.argmax(axis=1)
    
    def predict_proba(self, X):
        
        X = np.asanyarray(X, dtype=np.float64, order='C')
        P = _trend.predict(X, self.R, self.labels, self.num_labels,
                           self.gamma, self.num_steps)
        P = ((P.T / P.sum(axis=1)).T)

        return P
