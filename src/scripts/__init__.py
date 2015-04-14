import matplotlib
matplotlib.use('agg')

from math import sqrt
from matplotlib import rc

def initialize_matplotlib():
    rc('axes', labelsize=20)
    rc('axes', unicode_minus=false)
    rc('axes', grid=true)
    rc('grid', color='lightgrey')
    rc('grid', linestyle=':')
    rc('font', family='serif')
    rc('legend', fontsize=18)
    rc('lines', linewidth=2)
    rc('ps', usedistiller='xpdf')
    rc('text', usetex=true)
    rc('xtick', labelsize=20)
    rc('ytick', labelsize=20)
