#!/usr/bin/env python
# -*- coding: utf8

from __future__ import division, print_function

from learn_base import create_grid_search

from sklearn import base
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import tree

import numpy as np

class StackingException(Exception): pass

class Stacking(object):
    '''Implements a stacking classifier'''

    def __init__(self, num_splits, base_models, stacker_name='linear'):

        STACKERS = {'tree':_TreeStacking, 
                    'linear':_MLRStacking,
                    'deco':_DecoStacking}

        self.num_splits = num_splits
        self.base_classifiers = []

        for base_model in base_models:
            clone = base.clone(base_model)
            self.base_classifiers.append(clone)
        
        if stacker_name not in STACKERS:
            names = STACKERS.keys()
            raise StackingException('Unknown combiner, choose from: %s' % names)

        self.stacker = STACKERS[stacker_name]()
        self.P_fit = None
        self.y_fit = None
        self.model = None
        self.num_classes = 0

    def fit(self, X, y, B):
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        assert X.shape[0] == y.shape[0]
        assert y.ndim == 1

        self.num_classes = len(set(y))
        num_base_models = len(self.base_classifiers)

        P = np.zeros((X.shape[0], self.num_classes * num_base_models))

        kfold = cross_validation.StratifiedKFold(y, self.num_splits)
        for train, test in kfold:
            for i, base_model in enumerate(self.base_classifiers):
                base_model.fit(X[train], y[train])

                base_probs = base_model.predict_proba(X[test])
                cols_min = i * self.num_classes
                cols_max = self.num_classes * (i + 1)
                P[test, cols_min:cols_max] = base_probs

        #a = P.max(axis=1)
        #b = P.argmax(axis=1)
        #c = B.max(axis=1)
        #d = B.argmax(axis=1)
        #self.stacker.fit(np.vstack((a, b, c, d)).T, y)
        self.stacker.fit(np.hstack((P, B)), y)

    def predict(self, X, B):
        X = np.asanyarray(X)

        num_features = len(self.base_classifiers) * self.num_classes
        P = np.zeros((X.shape[0], num_features))
        for i, base_model in enumerate(self.base_classifiers):
            base_probs = base_model.predict_proba(X)
            cols_min = i * self.num_classes
            cols_max = self.num_classes * (i + 1)
            P[:, cols_min:cols_max] = base_probs

        #a = P.max(axis=1)
        #b = P.argmax(axis=1)
        #c = B.max(axis=1)
        #d = B.argmax(axis=1)
        #P = np.vstack((a, b, c, d)).T
        P = np.hstack((P, B))
        return self.stacker.predict(P)

class _MLRStacking(base.BaseEstimator, base.ClassifierMixin):
    """Implements a multi-response linear regression classifier"""

    def __init__(self):
        self.regressors = dict()

    def fit(self, X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        for yi in set(y):
            self.regressors[yi] = linear_model.LinearRegression()
            specific_y = np.asanyarray(y == yi, dtype='i')
            self.regressors[yi].fit(X, specific_y)

    def predict(self, X):
        X = np.asanyarray(X)
        
        prediction = np.zeros(X.shape[0])
        best_value = np.zeros_like(prediction)
        for yi, regressor in self.regressors.items():
            value = regressor.predict(X)
            for index, vindex in enumerate(value):
                if vindex > best_value[index]:
                    best_value[index] = vindex
                    prediction[index] = yi
        return prediction

class _TreeStacking(base.BaseEstimator, base.ClassifierMixin):
    '''Implements stacking with a multiresponse regression tree'''

    def __init__(self):
        self.model = None

    def _y_to_one_zero_mat(self, y):
        y = np.asanyarray(y)

        #Guarantees that y is 0 to n - 1
        unique_y, labels_flat = np.unique(y, return_inverse=True)
        y = labels_flat.reshape(y.shape)

        Y = np.zeros(shape=(len(y), len(unique_y)), dtype='f', order='C')
        for yi in unique_y:
            Y[:, yi] = (y == yi)

        return Y

    def fit(self, X, y):
        X = np.asanyarray(X, dtype='f', order='C')
        Y = self._y_to_one_zero_mat(y)
        
        self.model = tree.DecisionTreeRegressor()
        self.model.fit(X, Y)

    def predict(self, X):
        X = np.asanyarray(X, dtype='f', order='C')
        P = self.model.predict(X)
        return P.argmax(axis=1)

class _DecoStacking(base.BaseEstimator, base.ClassifierMixin):

    def __init__(self):
        self.model = etree = create_grid_search('extra_trees', n_jobs = 1)

    def fit(self, X, y):
        X = np.asanyarray(X, dtype='f', order='C')
        y = np.asanyarray(y, dtype='f', order='C')

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
