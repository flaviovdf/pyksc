#-*- coding: utf8

import _trend

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

import numpy as np

class TrendLearner(BaseEstimator, ClassifierMixin):

    def __init__(self, num_steps, gamma=1):
        self.num_steps = num_steps
        self.gamma = gamma
        self.num_labels = 0
        self.R = None
        self.labels = None

    def fit(self, X, y):

        self.R = np.asanyarray(X, dtype=np.float64, order='C')
        
        y = np.asanyarray(y)
        unique, labels_flat = np.unique(y, return_inverse=True)
        self.labels = labels_flat.reshape(y.shape)
        self.num_labels = unique.shape[0]
        

    def predict(self, X):

        X = np.asanyarray(X)[:, :self.num_steps]
        X = np.asanyarray(X, dtype=np.float64, order='C')
        
        P = _trend.predict(X, self.R, self.labels, self.num_labels, self.gamma)
        
        return P.argmax(axis=1)
    
    def predict_proba(self, X):
        
        X = np.asanyarray(X)[:, :self.num_steps]
        X = np.asanyarray(X, dtype=np.float64, order='C')
        
        P = _trend.predict(X, self.R, self.labels, self.num_labels, self.gamma)
        P = ((P.T / P.sum(axis=1)).T)

        return P
