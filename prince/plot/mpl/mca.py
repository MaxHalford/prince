import matplotlib.pyplot as plt

from .. import MCAPlotter
from ..palettes import GRAYS
from ..palettes import SEABORN
from . import MplPlotter
from . import util as mpl_util
from .pca import MplPCAPlotter


class MplMCAPlotter(MplPlotter, MCAPlotter):

    """Matplotlib plotter for Multiple Correspondence Analysis"""

    row_principal_coordinates = MplPCAPlotter.row_principal_coordinates

    def row_column_principal_coordinates(self, axes, row_principal_coordinates,
                                         column_principal_coordinates, explained_inertia,
                                         show_row_points, show_row_labels, show_column_points,
                                         show_column_labels):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.axhline(y=0, linestyle='-', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)
        ax.axvline(x=0, linestyle='-', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)

        if show_row_points:
            ax.plot(
                row_principal_coordinates.iloc[:, axes[0]],
                row_principal_coordinates.iloc[:, axes[1]],
                marker='+',
                linestyle='',
                ms=7,
                label='Row principal coordinates',
                color=GRAYS['dark']
            )

        if show_row_labels:
            for _, row in row_principal_coordinates.iterrows():
                ax.annotate(row.name, (row[axes[0]], row[axes[1]]))

        # Extract the prefixes from each column name for coloring
        if show_column_points or show_column_labels:
            data = column_principal_coordinates.iloc[:, axes].copy()
            data.columns = ('X', 'Y')
            # Only keep the prefix of each label
            data['label'] = column_principal_coordinates.index.to_series().apply(lambda x: x.split('_')[0])
            group_by = data.groupby('label')
            labels = sorted(group_by.groups.keys())
            n_colors = len(labels)
            cmap = mpl_util.create_discrete_cmap(n_colors)
            colors = cmap(range(n_colors))
            mpl_util.add_color_bar(ax, cmap, labels)

        if show_column_points:
            for (label, group), color in zip(group_by, colors):
                ax.scatter(group['X'], group['Y'], color=color, s=50, label=label)

        if show_column_labels:
            for (label, group), color in zip(group_by, colors):
                for _, row in group.iterrows():
                    ax.text(row['X'], row['Y'], s=row.name, color=color)

        ax.set_title('Row and column principal coordinates')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax

    def relationship_square(self, axes, column_correlations, explained_inertia, show_labels):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        data = column_correlations.iloc[:, axes].copy()
        data.columns = ('X', 'Y')

        for _, row in data.iterrows():
            ax.scatter(row['X'], row['Y'], s=50, color=SEABORN['blue'])
            if show_labels:
                ax.text(row['X'], row['Y'], s=row.name)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_title('Relationship square')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax
