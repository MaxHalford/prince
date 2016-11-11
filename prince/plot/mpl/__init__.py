import os

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .. import Plotter


class MplPlotter(Plotter):

    def __init__(self):

        self.qual_cmap = cm.Paired # Qualitative colormap
        self.colors = {
            'blue': '#0072b2',
            'green': '#009e73',
            'red': '#d55e00',
            'cyan': '#56b4e9',
            'light-gray': '#bababa',
            'dark-gray': '#404040'
        }

        mpl.rcParams['lines.linewidth'] = 1.8 # line width in points
        mpl.rcParams['lines.markeredgewidth'] = 0.3 # the line width around the marker symbol
        mpl.rcParams['lines.markersize'] = 7  # markersize, in points

        mpl.rcParams['grid.alpha'] = 0.4 # transparency, between 0.0 and 1.0



    def inertia(self, explained_inertia):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.plot(explained_inertia, color=self.colors['blue'], label='Normalized inertia')
        ax.plot(explained_inertia, 'o', color=self.colors['cyan'])

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

        # Threshold
        ax.axhline(y=threshold, color=self.colors['red'], label='Threshold',
                   linestyle='--')

        # First value above threshold
        try:
            index_above_threshold = [
                i >= threshold
                for i in cumulative_explained_inertia
            ].index(True)
            ax.axvline(x=index_above_threshold, color=self.colors['green'],
                       label='First component above threshold',
                       linestyle='--')
        except ValueError:
            pass

        # Inertia percentages
        ax.plot(cumulative_explained_inertia, color=self.colors['blue'],
                label='Normalized cumulative inertia')
        ax.plot(cumulative_explained_inertia, 'o', color=self.colors['cyan'])

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.margins(0.05, 0.15)
        ax.set_ylim(ymin=0)

        ax.set_title('Cumulative component contributions to inertia')
        ax.set_xlabel('Component number')
        ax.set_ylabel('Normalized cumulative inertia')
        ax.legend(loc='best')

        return fig, ax
