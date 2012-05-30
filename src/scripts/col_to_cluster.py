# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict
from matplotlib import pyplot as plt

from radar import radar_factory
from scipy import stats

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
            sum_classes[class_num] += 1
            to_plot[class_num][data] += 1
            
    return to_plot, sum_classes, sorted(labels)

def load_svm_file(features_fpath, classes):
    
    col_dict = {
        'EXTERNAL':13,
        'FEATURED':14,
        'INTERNAL':15,
        'MOBILE':16,
        'SEARCH':17,
        'SOCIAL':18,
        'VIRAL':19
    }

    to_plot = defaultdict(lambda: defaultdict(float))
    sum_classes = defaultdict(float)
    labels = set()
    with open(features_fpath) as features_file:
        curr_line = 0
        for line in features_file:
            if '#' in line:
                for key, id_ in col_dict.items():
                    print(id_, key, line.split()[id_])
                continue
            
            class_num = classes[curr_line]
            sum_classes[class_num] += float(line.split()[-1])
            for ref_name, col_id in col_dict.items():
                ref_abbrv = REFERRER_ABBRV[ref_name]
                
                val = float(line.split()[col_id])
                present = val > 0
                if present:
                    labels.add(ref_abbrv)
                    to_plot[class_num][ref_abbrv] += val
    
            curr_line += 1
            
    return to_plot, sum_classes, sorted(labels)

def generate_data_plot(to_plot, sum_classes, labels, classes):
    num_classes = len(set(classes))
    colors = ['b', 'g', 'm', 'y']
    
    total = 0        
    for class_num in xrange(num_classes):
        color = colors[class_num]
        
        data_plot = []
        for label in labels:
            total += to_plot[class_num][label]
            data_plot.append(to_plot[class_num][label] / sum_classes[class_num])
        
        yield data_plot, color, class_num
        
def radar_plot(labels, data_plots, out_fpath):

    theta = radar_factory(len(labels))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='radar')

    for data_plot, color, class_num in data_plots:
        ax.plot(theta, data_plot, color=color, label='C%d'%class_num)
        ax.fill(theta, data_plot, facecolor=color, alpha=0.25)
        
    ax.set_varlabels(labels)
    plt.legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.15), 
               loc='lower center')
    plt.savefig(out_fpath)

def chisq(counts, expected_prob):
    counts = np.array(counts)
    expected = np.array(expected_prob) * counts.sum()

    return stats.chisquare(counts, expected)[1]

def allchisq(to_plot, sum_classes, labels, classes):
    num_classes = len(set(classes))
    
    totals = []
    for label in labels:
        sum_ = 0
        for class_num in xrange(num_classes):
            sum_ += to_plot[class_num][label]
        totals.append(sum_)

    probs = []
    sum_totals = sum(totals)
    for i, t in enumerate(totals):
        probs.append( t / sum_totals)
    
    for class_num in xrange(num_classes):
        counts = []
        for label in labels:
            counts.append(to_plot[class_num][label])

        chisq(counts, probs)

def stacked_bars(labels, data_plots, out_fpath, label_translation, ref=True):
    
    x_locations = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]

    data_class = {}
    data_label = {}
    for data, _, class_num in data_plots:
        
        best_idx = np.argsort(data)[::-1][:4]
        best_cls = np.array(data)[best_idx] 
        best_lbl = np.array(labels)[best_idx]

        data_class[label_translation[class_num]] = best_cls
        data_label[label_translation[class_num]] = best_lbl

    bar_data   = []
    bar_labels = []
    for cls in sorted(data_class):
        bar_data.extend(data_class[cls])
        bar_labels.extend(data_label[cls])
    
    colors = ['b', 'g', 'm', 'r', 'y', 'c', '#A617A1', '#2B5700', 'w', 
              '#FF7300', 'k'] * 3

    colored={}
    if ref:
        to_use = set(REFERRER_ABBRV.values())
    else:
        to_use = set(CATEG_ABBRV.values())

    for i, l in enumerate(to_use):
        colored[l] = colors[i]

    for x, y, l in zip(x_locations, bar_data, bar_labels):
      
        c = colored[l]
        plt.bar(left=x, height=y, color=c, width=1, alpha=0.5)
        plt.text(x + .75, y, l, va='bottom', ha='center', rotation=45)
    
    plt.xlim(xmin=0, xmax=21)
    plt.xlabel('Cluster')
    if ref:
        plt.ylim(ymin=0, ymax=.31)
        plt.ylabel('Fraction of Views in Cluster')
    else:
        plt.ylim(ymin=0, ymax=.4)
        plt.ylabel('Fraction of Videos in Cluster')

    plt.xticks([3, 8, 13, 18],  ['$C0$', '$C1$', '$C2$', '$C3'])
    plt.savefig(out_fpath)

@plac.annotations(features_fpath=plac.Annotation('Features file', type=str),
                  classes_fpath=plac.Annotation('Video classes file', type=str),
                  out_fpath=plac.Annotation('Plot file', type=str),
                  trans_fpath=plac.Annotation('Translation of cluster num to labe', 
                                              type=str),
                  col_to_use=plac.Annotation('Column number to use', type=int,
                                             kind='option', abbrev='c'),
                  is_text_features=plac.Annotation('Indicates file type',
                                                   kind='flag', abbrev='t',
                                                   type=bool))
def main(features_fpath, classes_fpath, out_fpath, 
         trans_fpath, col_to_use=2, is_text_features=False):
    initialize_matplotlib()
    
    classes = np.loadtxt(classes_fpath)

    if is_text_features:
        to_plot, sum_classes, labels = \
        load_text_file(features_fpath, col_to_use, classes)
        ref=False
    else:
        to_plot, sum_classes, labels = \
        load_svm_file(features_fpath, classes)
        ref=True

    trans = {}
    with open(trans_fpath) as f:
        for l in f:
            spl = l.split()
            trans[int(spl[0])] = int(spl[1])

    data = generate_data_plot(to_plot, sum_classes, labels, classes)
    stacked_bars(labels, data, out_fpath, trans, ref)
    #allchisq(to_plot, sum_classes, labels, classes)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))
