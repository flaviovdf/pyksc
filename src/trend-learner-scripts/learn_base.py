#-*- coding: utf8
'''
Common functions for creating classifiers and regressors for machine learning
tasks
'''
from __future__ import division, print_function

from scipy import sparse

from sklearn import neighbors
from sklearn import ensemble
from sklearn import model_selection
from sklearn import linear_model
from sklearn import svm

import cStringIO
import numpy as np

#Params
TREE_SPLIT_RANGE = [1, 2, 4, 8, 16, 32, 64, 128]
KNN_K_RANGE = [5, 10, 15]

PARAMS = {'lr':{'C':[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]},
          'knn':{'n_neighbors':KNN_K_RANGE},
          'extra_trees':{'min_samples_split':TREE_SPLIT_RANGE}}

#Classifiers
CLFS = {'lr':linear_model.LogisticRegression(),
        'knn':neighbors.KNeighborsClassifier(),
        'extra_trees':ensemble.ExtraTreesClassifier(n_estimators=100,
                                                    criterion='entropy',
                                                    n_jobs=1)}

#Category Parsing Utilities
CATEG_ABBRV = {
    'Autos&amp;Vehicles':'Vehi.',
    'Autos':'Vehi.',
    'Comedy':'Com.',
    'Education':'Edu.',
    'Entertainment':'Ent.',
    'Film':'Film',
    'Film&amp;Animation':'Film',
    'Games':'Game',
    'Gaming':'Game',
    'Howto':'Howto',
    'Howto&amp;Style':'Howto',
    'Movies':'Film',
    'Music':'Music',
    'NULL':'-',
    'News':'News',
    'News&amp;Politics':'News',
    'Nonprofit':'Nonprof.',
    'Nonprofits&amp;Activism':'Nonprof.',
    'People&amp;Blogs':'People',
    'People':'People',
    'Pets&amp;Animals':'Pets',
    'Pets':'Pets',
    'Animals':'Pets',
    'Science&amp;Technology':'Sci.',
    'Science':'Sci.',
    'Tech':'Sci.',
    'Shows':'Show',
    'Sports':'Sport',
    'Trailers':'Film',
    'Travel&amp;Events':'Travel',
    'Travel':'Travel'}

CAT_COL = 2
CAT_IDS = dict((abbrv, i) \
               for i, abbrv in enumerate(sorted(set(CATEG_ABBRV.values()))))
INV_CAT_IDS = dict((v, k) for k, v in CAT_IDS.items())

def _get_classifier_and_params(name):
    return CLFS[name], PARAMS[name]

def create_grid_search(name, n_jobs=-1):
    learner, params = _get_classifier_and_params(name)    
    return model_selection.GridSearchCV(learner, params, cv=3, refit=True, 
                                    n_jobs=n_jobs)

def hstack_if_possible(X, Y):
    if X is not None:
        return np.hstack((X, Y))
    else:
        return Y
    
def load_categories(tags_cat_fpath):
    with open(tags_cat_fpath) as tags_cat_file:

        data = []
        for i, line in enumerate(tags_cat_file):
            spl = line.split()
            category = 'NULL'
            if len(spl) > CAT_COL:
                category = line.split()[CAT_COL]
                
            abbrv = CATEG_ABBRV[category]
            categ_id = CAT_IDS[abbrv]
            
            n_rows = len(CAT_IDS)
            row = np.zeros(n_rows)
            row[categ_id] = 1

            data.append(row)

        X_categ = np.asarray(data)
        return X_categ
            
def clf_summary(mean_scores, ci_scores):
    
    buff = cStringIO.StringIO()
    try:
        print('class \tprecision \trecall \tf1 score \tsupport', file=buff)
        for j in xrange(mean_scores.shape[1]):
            print(j, end="\t", file=buff)
            for i in xrange(mean_scores.shape[0]):
                print('%.3f +- %.3f' % (mean_scores[i, j], ci_scores[i, j]), 
                      end="\t", file=buff)
            print(file=buff)
        print(file=buff)
    
        return buff.getvalue()
    finally:
        buff.close()
