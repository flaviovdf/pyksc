# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict
from vod.entropy import kullback_leiber_divergence

import numpy as np
import plac
import sys

def load_text_file(features_fpath, classes, use):
    #TODO: stemming and category names abbrv
    num_classes = len(set(classes))
    
    count_class = [0] * num_classes
    prob_col = defaultdict(float)
    count_class_col = defaultdict(lambda: defaultdict(float))
    
    with open(features_fpath) as features_file:
        for curr_line, line in enumerate(features_file):
            spl = line.split()
            
            class_num = classes[curr_line]
            
            if use == 'user':
                count_class_col[spl[1]][class_num] += 1
                prob_col[spl[1]] += 1
            elif use == 'cat':
                if len(spl) > 2:
                    count_class_col[spl[2]][class_num] += 1
                    prob_col[spl[2]] += 1
            else:
                for token in spl[3:]:
                    prob_col[token] += 1
                    count_class_col[token][class_num] += 1
            
            count_class[int(class_num)] += 1
    
    prob_class = np.array(count_class, dtype='f')
    prob_class /= prob_class.sum()
    
    prob_class_col = {}
    sum_col = sum(prob_col.values())
    for token in count_class_col:
        prob_col[token] = prob_col[token] / sum_col
         
        aux = np.zeros(num_classes, dtype='f')
        for class_num in xrange(num_classes):
            aux[class_num] = count_class_col[token][class_num]
        aux /= aux.sum()
        
        prob_class_col[token] = aux 
            
    return prob_class, prob_col, prob_class_col

def load_svm_file(features_fpath, classes):
    col_dict = {
        'EXTERNAL':8,
        'FEATURED':9,
        'INTERNAL':10,
        'MOBILE':11,
        'SEARCH':12,
        'SOCIAL':13,
        'VIRAL':14
    }

    num_classes = len(set(classes))
    count_class = [0] * num_classes
    prob_col = defaultdict(float)
    count_class_col = defaultdict(lambda: defaultdict(float))

    with open(features_fpath) as features_file:
        curr_line = 0
        for line in features_file:
            if '#' in line:
                continue
            
            spl = line.split()
            for ref_name, col_id in col_dict.items():
                ref_abbrv = ref_name
                class_num = classes[curr_line]
                
                weight = float(spl[col_id])
                
                prob_col[ref_abbrv] += weight
                count_class[int(class_num)] += 1
                count_class_col[ref_abbrv][class_num] += weight
    
            curr_line += 1
            
    prob_class = np.array(count_class, dtype='f')
    prob_class /= prob_class.sum()
    
    prob_class_col = {}
    sum_col = sum(prob_col.values())
    for token in count_class_col:
        prob_col[token] = prob_col[token] / sum_col
         
        aux = np.zeros(num_classes, dtype='f')
        for class_num in xrange(num_classes):
            aux[class_num] = count_class_col[token][class_num]
        aux /= aux.sum()
        
        prob_class_col[token] = aux 
            
    return prob_class, prob_col, prob_class_col

@plac.annotations(features_fpath=plac.Annotation('Input file', type=str),
                  classes_fpath=plac.Annotation('Video classes file', type=str),
                  use=plac.Annotation('Indicates which information to use',
                                      type=str, 
                                      choices=['user', 'tags', 'cat', 'ref']))
def main(features_fpath, classes_fpath, use):
    
    classes = np.loadtxt(classes_fpath)
    
    if use in {'user', 'tags', 'cat'}:
        prob_class, prob_col, prob_class_col = load_text_file(features_fpath, 
                                                              classes, use)
    else:
        prob_class, prob_col, prob_class_col = load_svm_file(features_fpath, 
                                                             classes)
    info_gains = []
    mutual_info = 0

    for token in prob_class_col:
        dkl = kullback_leiber_divergence(prob_class_col[token], prob_class)
        
        mutual_info += prob_col[token] * dkl
        info_gains.append((dkl, token))
    
    print('Mutual info: ', mutual_info)
    for dkl, token in sorted(info_gains, reverse=True):
        print(dkl, token)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))