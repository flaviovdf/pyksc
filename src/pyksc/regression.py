#-*- coding: utf8
'''
Implementation of some machine Learning regression models. Here we provide 
wrappers around scikit-learn regression classes.
'''
from __future__ import division, print_function

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin 
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import safe_asarray

import numpy as np

class RSELinearRegression(LinearRegression):
    '''
    Implements an ordinary least squares (OLS) linear regression in which
    the objective function is the relative squared error (RSE) and not the 
    absolute error.
    
    This class will use the same parameters and arguments as:
    sklearn.linear_model.LinearRegression
    
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
    
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        super(RSELinearRegression, self).__init__(fit_intercept, normalize,
                                                  copy_X)
        
    def fit(self, X, y):
        X = safe_asarray(X)
        y = np.asarray(y)
        
        X = (X.T / y).T
        return super(RSELinearRegression, self).fit(X, y)

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
    clf_class : a subclass of `sklearn.base.ClassifierMixin`
        this is a class object and not a instance of the class
    clf_params : dict
        the parameters for the classifier
    regression_class : a subclass of `sklearn.base.RegressorMixin`
        this is a class object and not a instance of the class
    regression_params: dict
        the parameters used for building the regression model
    '''
    
    def __init__(self, clf_class, clf_params, 
                 regression_class, regression_params):
        
        super(MultiClassRegression, self).__init__()
        
        self.clf_class = clf_class
        self.clf_params = clf_params
        self.regression_class = regression_class
        self.regression_params = regression_params
        
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
        
        self.clf_model = self.clf_class.fit(X, y_clf)
        
        classes = set(y_clf)
        self.regression_models = {}
        
        for class_ in classes:
            examples = y_clf == class_
            
            X_class = X[examples]
            y_class = y_regression[examples]
            self.regression_models[class_] = self.regression_class.fit(X_class, 
                                                                       y_class)
        
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
        y_clf_predicted = np.asarray(self.clf_model.predict(X))
        classes = set(y_clf_predicted)
        
        y_predicted = np.zeros(X.shape[0], dtype='f')
        for class_ in classes:
            examples = y_clf_predicted == class_
            regression_model = self.regression_models[class_]
            y_predicted[examples] = regression_model.predict(X[examples])  

        return y_predicted