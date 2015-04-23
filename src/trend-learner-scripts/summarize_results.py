#-*- coding: utf8
from __future__ import division, print_function

from pyksc import dist

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import glob
import numpy as np
import os
import plac 

def main(tseries_fpath, base_folder):

    folders = glob.glob(os.path.join(base_folder, 'fold-*'))
    num_folders = len(folders)
    
    cluster_mapping = []
    C_base = np.loadtxt(os.path.join(folders[0], 'ksc/cents.dat'))
    
    for i in xrange(num_folders):
        Ci = np.loadtxt(os.path.join(folders[i], 'ksc/cents.dat'))

        dists = dist.dist_all(Ci, C_base, rolling=True)[0]
        closest = dists.argmin(axis=1)
        
        cluster_mapping.append({})
        for k in xrange(Ci.shape[0]):
            cluster_mapping[i][k] = closest[k]
    
    y_true_all = []
    y_pred_all = []
    for i in xrange(num_folders):
        y_true = np.loadtxt(os.path.join(folders[i], 'ksc/test_assign.dat'))
        y_pred = np.loadtxt(os.path.join(folders[i], \
                'cls-res-fitted-50/pred.dat'))
        
        for j in xrange(y_true.shape[0]):
            y_true[j] = cluster_mapping[i][y_true[j]]
            if y_pred[j] != -1:
                y_pred[j] = cluster_mapping[i][y_pred[j]]
        
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    
    y_pred_all = np.asarray(y_pred_all)
    y_true_all = np.asarray(y_true_all)

    report = classification_report(y_true_all, y_pred_all)
    valid = y_pred_all != -1
    print()
    print('Using the centroids from folder: ', folders[0])
    print('Micro Aggregation of Folds:')
    print('%.3f fract of videos were not classified' % (sum(~valid) / y_pred_all.shape[0]))
    print()
    print(classification_report(y_true_all[valid], y_pred_all[valid]))

if __name__ == '__main__':
    plac.call(main)
