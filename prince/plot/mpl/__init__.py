import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .. import Plotter
from ..palettes import SEABORN


class MplPlotter(Plotter):

    def __init__(self):

        mpl.rcParams['figure.figsize'] = (9.6, 7.2)

        mpl.rcParams['lines.linewidth'] = 1.8 # line width in points
        mpl.rcParams['lines.markeredgewidth'] = 0.3 # the line width around the marker symbol
        mpl.rcParams['lines.markersize'] = 7  # markersize, in points

        mpl.rcParams['grid.alpha'] = 0.5 # transparency, between 0.0 and 1.0
        mpl.rcParams['grid.linestyle'] = '-' # simple line
        mpl.rcParams['grid.linewidth'] = 0.4 # in points

    def inertia(self, explained_inertia):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.plot(explained_inertia, color=SEABORN['blue'], label='Normalized inertia')
        ax.plot(explained_inertia, 'o', color=SEABORN['cyan'])

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.margins(0.05)
        ax.set_ylim(ymax=1)

        ax.set_title('Component contributions to inertia')
        ax.set_xlabel('Component number')
        ax.set_ylabel('Normalized inertia')
        ax.legend(loc='best')

        return fig, ax

    def cumulative_inertia(self, cumulative_explained_inertia, threshold):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Plot threshold line
        ax.axhline(y=threshold, color=SEABORN['red'], label='Threshold',
                   linestyle='--')

        # Plot first value above threshold line
        try:
            index_above_threshold = [
                i >= threshold
                for i in cumulative_explained_inertia
            ].index(True)
            ax.axvline(x=index_above_threshold, color=SEABORN['green'],
                       label='First component above threshold',
                       linestyle='--')
        except ValueError:
            pass

        # Plot inertia percentages curve
        ax.plot(cumulative_explained_inertia, color=SEABORN['blue'],
                label='Normalized cumulative inertia')
        ax.plot(cumulative_explained_inertia, 'o', color=SEABORN['blue'])

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.margins(0.05, 0.15)
        ax.set_ylim(ymin=0)

        ax.set_title('Cumulative component contributions to inertia')
        ax.set_xlabel('Component number')
        ax.set_ylabel('Normalized cumulative inertia')
        ax.legend(loc='best')

        return fig, ax
