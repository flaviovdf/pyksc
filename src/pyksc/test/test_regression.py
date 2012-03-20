#-*- coding: utf8
'''Unit tests for the regression module'''

from __future__ import division, print_function

from numpy.testing import *
from pyksc import regression

from sklearn import linear_model
from sklearn.grid_search import GridSearchCV

import numpy as np
import unittest


class TestRSELinearRegression(unittest.TestCase):
    
    def test_rse(self):
        assert_almost_equal(regression.mean_relative_square_error([1, 1, 1], 
                                                                  [0, 0, 0]), 1)
                
        assert_equal(regression.mean_relative_square_error([1, 0.5, 0.8], 
                                                           [1, 0.5, 0.8]), 0)

    def test_rse_fit_one_attr(self):
        
        X = [[1],
             [4]]
        
        X_conv = [[1],
                  [2]]
        y = [1, 2]
        
        rse_lsq = regression.RSELinearRegression(fit_intercept=False)
        lsq = linear_model.LinearRegression(fit_intercept=False)
        
        model_rse = rse_lsq.fit(X, y)
        model_lsq = lsq.fit(X_conv, np.ones(len(y)))
        
        assert_array_equal(model_lsq.coef_, model_rse.coef_)
        assert_equal(model_lsq.intercept_, model_rse.intercept_)
        
        assert_array_almost_equal(model_rse.predict([[1], [4]]),
                                  model_lsq.predict([[1], [4]]))
    
    def test_rse_fit(self):
        
        X = [[1.0, 2],
             [4, 8]]
        
        X_conv = [[1.0, 2],
                  [2, 4]]
        y = [1, 2]
        
        rse_lsq = regression.RSELinearRegression(fit_intercept=False)
        lsq = linear_model.LinearRegression(fit_intercept=False)
        
        model_rse = rse_lsq.fit(X, y)
        model_lsq = lsq.fit(X_conv, np.ones(len(y)))
        
        assert_array_equal(model_lsq.coef_, model_rse.coef_)
        assert_equal(model_lsq.intercept_, model_rse.intercept_)
        
        assert_array_almost_equal(model_rse.predict([[1, 2], [1, 2]]),
                                  model_lsq.predict([[1, 2], [1, 2]]))
        
class TestMultiClassRegression(unittest.TestCase):
    
    def test_multiclass(self):
        
        X = [[1, 2],
             [1, 2],
             [4, 8],
             [4, 8],
             [200, 200],
             [199, 200.1],
             [200.2, 198]]
        
        y_clf = [0, 0, 0, 0, 1, 1, 1]
        y_regression = [1, 1, 2, 2, 100, 100, 100]
        
        regr_class = regression.RSELinearRegression(fit_intercept=False)
        clf_class = linear_model.LogisticRegression()
        
        multi_class = regression.MultiClassRegression(clf_class, regr_class)
        
        model = multi_class.fit(X, y_clf, y_regression)
        p = model.predict([[1, 2], 
                           [200, 200], 
                           [1, 2], 
                           [200, 200]])
        assert_equal(p[0], p[2])
        assert_equal(p[1], p[3])
        self.assertTrue(p[0] != p[1])

    def test_with_grid_search(self):
        X = [[1, 2],
             [1, 2],
             [4, 8],
             [4, 8],
             [200, 200],
             [199, 200.1],
             [200.2, 198]]
        
        y_clf = [0, 0, 0, 0, 1, 1, 1]
        y_regression = [1, 1, 2, 2, 100, 100, 100]
        
        regr_class = GridSearchCV(regression.RSELinearRegression(), 
                                  {'normalize':[0,1]})
        clf_class = GridSearchCV(linear_model.LogisticRegression(), {'C':[1,2]})
        
        multi_class = regression.MultiClassRegression(clf_class, regr_class)
        
        model = multi_class.fit(X, y_clf, y_regression)
        p = model.predict([[1, 2], 
                           [200, 200], 
                           [1, 2], 
                           [200, 200]])
        
        assert_equal(p[0], p[2])
        assert_equal(p[1], p[3])
        self.assertTrue(p[0] != p[1])