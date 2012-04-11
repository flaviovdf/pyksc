# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict
from matplotlib import pyplot as plt
from radar import radar_factory
import rpy2.robjects as robjects
from scripts import initialize_matplotlib

import numpy as np
import plac
import sys

REFERRER_ABBRV = {
    'EXTERNAL':'EXT.',
    'FEATURED':'FEAT.',
    'INTERNAL':'INT.',
    'MOBILE':'MOBI.',
    'SEARCH':'SEAR.',
    'SOCIAL':'SOC.',
    'VIRAL':'VIR.'}

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
    'Music':'Mus.',
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
    'Travel&amp;Events':'Travel',
    'Travel':'Travel'}
    
def load_text_file(features_fpath, col_to_use, classes):
    
    to_plot = defaultdict(lambda: defaultdict(float))
    sum_classes = defaultdict(float)
    labels = set()
    with open(features_fpath) as features_file:
        for curr_line, line in enumerate(features_file):
            spl = line.split()
            if col_to_use >= len(spl):
                continue
            
            data = CATEG_ABBRV[line.split()[col_to_use].strip()]
            class_num = classes[curr_line]
            
            labels.add(data)
            
            sum_classes[data] += 1 
            to_plot[class_num][data] += 1
            
    return to_plot, sum_classes, sorted(labels)

def load_svm_file(features_fpath, classes):
    
    col_dict = {
        'EXTERNAL':1,
        'FEATURED':2,
        'INTERNAL':3,
        'MOBILE':4,
        'SEARCH':5,
        'SOCIAL':6,
        'VIRAL':7
    }

    to_plot = defaultdict(lambda: defaultdict(float))
    sum_classes = defaultdict(float)
    labels = set()
    with open(features_fpath) as features_file:
        curr_line = 0
        for line in features_file:
            if '#' in line:
                continue
            
            for ref_name, col_id in col_dict.items():
                ref_abbrv = REFERRER_ABBRV[ref_name]
                class_num = classes[curr_line]
                
                present = float(line.split()[col_id]) > 0
                if present:
                    labels.add(ref_abbrv)
                    sum_classes[ref_abbrv] += 1
                    to_plot[class_num][ref_abbrv] += 1
    
            curr_line += 1
            
    return to_plot, sum_classes, sorted(labels)

def generate_data_plot(to_plot, sum_classes, labels, classes, out_fpath):
    num_classes = len(set(classes))
    colors = ['b', 'g', 'm', 'y']
    
    total = 0        
    for class_num in xrange(num_classes):
        color = colors[class_num]
        num_members = sum(classes == class_num)
        
        data_plot = []
        f_obs = []
        f_exp = []
        for label in labels:
            total += to_plot[class_num][label]
            
            if to_plot[class_num][label] >= 5:
                f_obs.append(to_plot[class_num][label])
                f_exp.append(sum_classes[label])
                
            data_plot.append(to_plot[class_num][label] / num_members)
        
        yield data_plot, color, class_num
        
def radar_plot(labels, data_plots, out_fpath):

    theta = radar_factory(len(labels))
#    radial_grid = [0.05, 0.1, 0.15]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='radar')
#    ax.set_rlim(0.20)
#    ax.set_rmax(0.20)
#    plt.rgrids(radial_grid)

    for data_plot, color, class_num in data_plots:
        ax.plot(theta, data_plot, color=color, label='C%d'%class_num)
        ax.fill(theta, data_plot, facecolor=color, alpha=0.25)
        
    ax.set_varlabels(labels)
    plt.legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.125), 
               loc='lower center')
    plt.savefig(out_fpath)


def stacked_bars(labels, data_plots, out_fpath):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
#    bars = []
#    colors = []
#    class_nums = []
    for data_plot, color, class_num in data_plots:
        
        bottom = np.arange(1.5, len(data_plot)*3 + 0.6, 3)
        ax.barh(bottom = bottom, width = data_plot, height=2.5, 
                label ='C%d'%class_num, color = color, alpha=0.25)
        
#        colors.append(color)
#        class_nums.append(class_num)
#        
#        i = 0
#        for data in data_plot:
#            bars[i].append(data)
#            i += 1
#
#    for i, class_num in enumerate(class_nums):
    plt.savefig(out_fpath)

@plac.annotations(features_fpath=plac.Annotation('Features file', type=str),
                  classes_fpath=plac.Annotation('Video classes file', type=str),
                  out_fpath=plac.Annotation('Video classes file', type=str),
                  col_to_use=plac.Annotation('Column number to use', type=int,
                                             kind='option', abbrev='c'),
                  is_text_features=plac.Annotation('Indicates file type',
                                                   kind='flag', abbrev='t',
                                                   type=bool))
def main(features_fpath, classes_fpath, out_fpath, 
         col_to_use=0, is_text_features=False):
    initialize_matplotlib()
    
    classes = np.loadtxt(classes_fpath)
    
    if is_text_features:
        to_plot, sum_classes, labels = \
        load_text_file(features_fpath, col_to_use, classes)
    else:
        to_plot, sum_classes, labels = \
        load_svm_file(features_fpath, classes)

    data = generate_data_plot(to_plot, sum_classes, labels, classes, out_fpath)
    radar_plot(labels, data, out_fpath)
#    stacked_bars(labels, data, out_fpath)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))