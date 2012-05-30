# -*- coding: utf8

from __future__ import division, print_function

from scripts.learn_base import create_input_table, hstack_if_possible

import numpy as np
import plac
import sys

@plac.annotations(features_fpath=plac.Annotation('Partial Features', 
                                                         type=str),
                  tag_categ_fpath=plac.Annotation('Tags file', type=str),
                  tseries_fpath=plac.Annotation('Time series file', type=str),
                  assign_fpath=plac.Annotation('Series assignment file', 
                                               type=str))
def main(features_fpath, tag_categ_fpath, tseries_fpath, assign_fpath):
    
    X, feature_ids, _ = \
            create_input_table(features_fpath, None, tag_categ_fpath,-1)
   
    y_clf = np.genfromtxt(assign_fpath)
    y_rgr = np.genfromtxt(tseries_fpath)[:,1:].sum(axis=1)

    for feat_id in range(len(feature_ids)):
        print(feature_ids[feat_id], end=',')
    
    print('TREND', end=',')
    print('FINAL_VIEWS')
    
    M = np.column_stack((X, y_clf, y_rgr))
    np.savetxt(sys.stdout, M, '%d', delimiter=',')

if __name__ == '__main__':
    sys.exit(plac.call(main))