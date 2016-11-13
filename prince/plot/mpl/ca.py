import matplotlib.pyplot as plt

from .. import CAPlotter
from ..palettes import GRAYS
from ..palettes import SEABORN
from . import MplPlotter


class MplCAPlotter(MplPlotter, CAPlotter):

    """Matplotlib plotter for Correspondence Analysis"""

    def row_column_projections(self, row_projections, column_projections, axes, explained_inertia,
                               show_row_points, show_row_labels, show_column_points,
                               show_column_labels):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.axhline(y=0, linestyle='-', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)
        ax.axvline(x=0, linestyle='-', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)

        row_pc = row_projections.iloc[:, axes].copy()
        row_pc.columns = ('X', 'Y')
        col_pc = column_projections.iloc[:, axes].copy()
        col_pc.columns = ('X', 'Y')

        if show_row_points:
            ax.scatter(row_pc['X'], row_pc['Y'], c=SEABORN['blue'], s=50,
                       label='Row projections')

        if show_row_labels:
            ax.scatter(row_pc['X'], row_pc['Y'], alpha=0,
                       label=None if show_row_points else 'Row projections')
            for _, row in row_projections.iterrows():
                ax.annotate(row.name, (row[axes[0]], row[axes[1]]), color=SEABORN['blue'])

        if show_column_points:
            ax.scatter(col_pc['X'], col_pc['Y'], c=SEABORN['red'], s=50,
                       label='Column projections')

        if show_column_labels:
            ax.scatter(col_pc['X'], col_pc['Y'], alpha=0,
                       label=None if show_column_points else 'Column projections')
            for _, row in column_projections.iterrows():
                ax.annotate(row.name, (row[axes[0]], row[axes[1]]), color=SEABORN['red'])

        ax.axis('equal')

        ax.set_title('Row and column projections')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))
        if show_row_points or show_column_points:
            ax.legend(loc='best')

        return fig, ax
