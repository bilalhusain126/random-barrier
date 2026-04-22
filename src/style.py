"""Publication-quality plot styling: STIX fonts, no grids, clean spines."""
import matplotlib.pyplot as plt
import logging

logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)


def setup_publication_style():
    plt.rcParams.update({
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'stix',

        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 13,

        'axes.linewidth': 0.8,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,

        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
