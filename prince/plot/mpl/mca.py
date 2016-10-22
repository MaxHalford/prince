import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .. import MCAPlotter
from .pca import MplPCAPlotter
from . import MplPlotter


class MplMCAPlotter(MplPlotter, MCAPlotter):

    """Matplotlib plotter for Multiple Correspondence Analysis"""

    row_projections = MplPCAPlotter.__dict__['row_projections']

    def row_column_projections(self, axes, row_projections, column_projections, explained_inertia,
                               show_row_points, show_row_labels, show_column_points,
                               show_column_labels):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.axhline(y=0, linestyle='-', linewidth=1.2, color=self.colors['dark-gray'], alpha=0.6)
        ax.axvline(x=0, linestyle='-', linewidth=1.2, color=self.colors['dark-gray'], alpha=0.6)

        if show_row_points:
            ax.plot(row_projections.iloc[:, axes[0]], row_projections.iloc[:, axes[1]], marker='+',
                    linestyle='', ms=7, label='Row projections', color=self.colors['dark-gray'])

        if show_row_labels:
            for _, row in row_projections.iterrows():
                ax.annotate(row.name, (row[axes[0]], row[axes[1]]))

        # Extract the prefixes from each variable name for coloring
        if show_column_points or show_column_labels:
            data = column_projections.iloc[:, axes].copy()
            data.columns = ('X', 'Y')
            # Only keep the prefix of each label
            labels = column_projections.index.to_series().apply(lambda x: x.split('_')[0])
            data['label'] = labels
            groups = data.groupby('label')
            colors = self.qual_cmap(np.linspace(0, 1, len(groups)))
            # Draw a colorbar
            cax = fig.add_axes((0.9, 0.1, 0.03, 0.8))
            bounds = list(range(len(groups)+1))
            norm = mpl.colors.BoundaryNorm(bounds, self.qual_cmap.N)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=self.qual_cmap, norm=norm, ticks=bounds)
            cbar.ax.set_yticklabels(data['label'].unique())

        if show_column_points:
            for (label, group), color in zip(groups, colors):
                ax.scatter(group['X'], group['Y'], color=color, s=50, label=label)

        if show_column_labels:
            for (label, group), color in zip(groups, colors):
                for _, row in group.iterrows():
                    ax.text(row['X'], row['Y'], s=row.name, color=color)

        ax.set_title('Row and column projections')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax

    def relationship_square(self, axes, variable_correlations, explained_inertia, show_labels):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        data = variable_correlations.iloc[:, axes].copy()
        data.columns = ('X', 'Y')

        for _, row in data.iterrows():
            ax.scatter(row['X'], row['Y'], s=50, color=self.colors['blue'])
            if show_labels:
                ax.text(row['X'], row['Y'], s=row.name)

        ax.set_title('Relationship square')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax
