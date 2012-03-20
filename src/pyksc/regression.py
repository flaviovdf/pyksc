#-*- coding: utf8
'''
Implementation of some Machine Learning regression models. Basically, we 
implement simple wrappers around the scikit-learn library which performs
the transformations and specific training models we need.
'''
from __future__ import division, print_function

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin 
from sklearn.linear_model.base import LinearRegression
from sklearn.utils.validation import safe_asarray

import numpy as np

def mean_relative_square_error(y_true, y_pred):
    """
    Mean relative square error regression loss

    Positive floating point value: the best value is 0.0.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    mrse : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_pred / y_true - 1) ** 2)
    
class RSELinearRegression(LinearRegression):
    '''
    Implements an ordinary least squares (OLS) linear regression in which
    the objective function is the relative squared error (RSE) and not the 
    absolute error.
    
    This class will use the same parameters and arguments as:
    sklearn.linear_model.LinearRegression. Different from the linear
    regression, we set `fit_intecept` to False by default.
    
    Parameters
    ----------
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean, optional
        If True, the regressors X are normalized
    
    See
    ---
    sklearn.linear_model.LinearRegression
    '''
    
    def __init__(self, fit_intercept=False, normalize=False, copy_X=True):
        super(RSELinearRegression, self).__init__(fit_intercept, normalize, 
                                                  copy_X)
        
    def fit(self, X, y):
        X = safe_asarray(X)
        y = np.asarray(y)
        
        X = (X.T / y).T
        return super(RSELinearRegression, self).fit(X, np.ones(len(y)))

    def score(self, X, y):
        return mean_relative_square_error(y, self.predict(X))

class MultiClassRegression(BaseEstimator, RegressorMixin):
    '''
    This class implements what we call a multi-class regression. In simple
    terms, for a dataset with class labels one specialized regression model
    is learned for each label. Also, a classification model is learned for the
    whole dataset. Thus, when predicting first the classification model is used
    to infer classes and secondly the specialized regression model for each
    class is used.

    Parameters
    ----------
    clf : an instance of `sklearn.base.ClassifierMixin`
        this is the classifier to be used. Pass a grid search object when
        searching for best parameters is needed
    regr : a subclass of `sklearn.base.RegressorMixin`
        this is a class object and not a instance of the class. Pass a grid 
        search object when searching for best parameters is needed
    '''
    
    def __init__(self, clf, regr):
        super(MultiClassRegression, self).__init__()
        
        self.clf = clf
        self.regr = regr
        self.clf_model = None
        self.regression_models = None
    
    def fit(self, X, y_clf, y_regression):
        """
        Fit the multiclass model.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y_clf : numpy array of shape [n_samples]
            Target classes for classification model
        y_regression: numpy array of shape [n_samples]
            Target values for regression model 
            
        Returns
        -------
        self : returns an instance of self.
        """
        
        X = safe_asarray(X)
        y_clf = np.asarray(y_clf)
        y_regression = np.asarray(y_regression)
        
        self.clf_model = self.clf.fit(X, y_clf)
        
        classes = set(y_clf)
        self.regression_models = {}
        
        #TODO: check if we can use scikits Parallel class for parallel fit
        for class_ in classes:
            examples = y_clf == class_
            self.regression_models[class_] = \
                    clone(self.regr).fit(X[examples], y_regression[examples])
        
        return self

    def predict(self, X):
        """
        Predict using the muticlass regression model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        
        X = safe_asarray(X)
        y_clf_predicted = np.asarray(self.clf_model.predict(X))
        classes = set(y_clf_predicted)
        
        #TODO: check if we can use scikits Parallel class for parallel fit
        y_predicted = None
        for class_ in classes:
            examples = y_clf_predicted == class_
            regression_model = self.regression_models[class_]
            
            regression_prediction = regression_model.predict(X[examples])
            if y_predicted is None:
                y_predicted = np.zeros(X.shape[0], regression_prediction.dtype)
            
            y_predicted[examples] = regression_model.predict(X[examples])

        return y_predicted