# -*- coding: utf8

from __future__ import division, print_function

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import scale

import argparse
import numpy as np
import sys
import traceback

def main(features_fpath, classes_fpath):
    
    with open(features_fpath) as features_file:
        for line in features_file:
            if '#' in line:
                spl = line.split()
                names = spl[1:]
    
    X = scale(np.genfromtxt(features_fpath)[:,1:].copy())
    y = np.loadtxt(classes_fpath)
    
    forest = ExtraTreesClassifier(n_estimators=250,
                                  compute_importances=True)
    
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    for f in xrange(10):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], 
                                       importances[indices[f]]))

def create_parser(prog_name):
    
    desc = __doc__
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog_name, description=desc, 
                                     formatter_class=formatter)
    
    parser.add_argument('features_fpath', type=str,
                        help='Input file with video features')
    parser.add_argument('classes_fpath', type=str,
                        help='Classes to predict')
    
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.features_fpath, values.classes_fpath)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))