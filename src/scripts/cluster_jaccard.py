# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict
from scripts import initialize_matplotlib

import numpy as np
import plac
import sys

def load_text_file(features_fpath, classes, user_users):
    
    to_cmp = defaultdict(set)
    
    with open(features_fpath) as features_file:
        for curr_line, line in enumerate(features_file):
            spl = line.split()
            
            if user_users:
                data = set([spl[1]])
            else:
                data = set(token.strip() for token in spl[2:])
            
            class_num = classes[curr_line]
            to_cmp[class_num].update(data)
            
    return to_cmp

def asym_jaccard(first_set, second_set):
    intersect = first_set.intersection(second_set)
    return len(intersect) / len(first_set)

@plac.annotations(features_fpath=plac.Annotation('Tags file', type=str),
                  classes_fpath=plac.Annotation('Video classes file', type=str),
                  user_users=plac.Annotation('Use user_names instead of tags',
                                                   kind='flag', abbrev='u',
                                                   type=bool))
def main(features_fpath, classes_fpath, user_users=False):
    
    initialize_matplotlib()
    
    classes = np.loadtxt(classes_fpath)
    num_classes = len(set(classes))
    
    to_compare = load_text_file(features_fpath, classes, user_users)
    
    print(end='\t')
    for i in xrange(num_classes):
        print(i, end='\t')
    print()
    
    for j in xrange(num_classes):
        print(j, end='\t')
        for i in xrange(num_classes):
            
            first_set = to_compare[i]
            second_set = to_compare[j]
            
            asym_j = asym_jaccard(first_set, second_set)
            print('%.3f' % asym_j, end='\t')
        print()
    
if __name__ == '__main__':
    sys.exit(plac.call(main))