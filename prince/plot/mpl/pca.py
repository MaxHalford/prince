import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .. import PCAPlotter
from .. import util as plot_util
from ..palettes import GRAYS
from ..palettes import SEABORN
from . import MplPlotter
from . import util as mpl_util


class MplPCAPlotter(MplPlotter, PCAPlotter):

    """Matplotlib plotter for Principal Component Analysis"""

    def row_principal_coordinates(self, axes, principal_coordinates,
                                  supplementary_principal_coordinates, explained_inertia,
                                  show_points, show_labels, color_labels, ellipse_outline,
                                  ellipse_fill):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.axhline(y=0, linestyle='-', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)
        ax.axvline(x=0, linestyle='-', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)

        data = principal_coordinates.iloc[:, axes].copy() # Active rows
        supp = supplementary_principal_coordinates.iloc[:, axes].copy() # Supplementary rows
        data.columns = ('X', 'Y')
        supp.columns = ('X', 'Y')

        # Choose colors and add corresponding colorbar if necessary
        if (color_labels is not None) and (show_points or show_labels or ellipse_outline or ellipse_fill):
            data['label'] = color_labels
            supp['label'] = color_labels
            group_by = data.groupby('label')
            if not supplementary_principal_coordinates.empty:
                group_by_supp = supp.groupby('label')
            labels = sorted(group_by.groups.keys())
            n_colors = len(labels)
            cmap = mpl_util.create_discrete_cmap(n_colors)
            colors = cmap(range(n_colors))
            mpl_util.add_color_bar(ax, cmap, labels)

        if show_points:
            if color_labels is not None:
                for (label, group), color in zip(group_by, colors):
                    ax.scatter(group['X'], group['Y'], s=50, color=color, label=label)
                if not supplementary_principal_coordinates.empty:
                    for (label, group), color in zip(group_by_supp, colors):
                        ax.scatter(group['X'], group['Y'], s=90, color=color, label=label,
                                   marker='*')
            else:
                ax.scatter(data['X'], data['Y'], s=50, color=SEABORN['blue'])

        if show_labels:
            ax.scatter(data['X'], data['Y'], alpha=0, label=None)
            if color_labels is not None:
                for (label, group), color in zip(group_by, colors):
                    for _, row in group.iterrows():
                        ax.text(x=row['X'], y=row['Y'], s=row.name, color=color)
                if not supplementary_principal_coordinates.empty:
                    for (label, group), color in zip(group_by_supp, colors):
                        for _, row in group.iterrows():
                            ax.text(x=row['X'], y=row['Y'], s=row.name, color=color)
            else:
                for _, row in data.iterrows():
                    ax.text(x=row['X'], y=row['Y'], s=row.name)
                if not supplementary_principal_coordinates.empty:
                    for _, row in supp.iterrows():
                        ax.text(x=row['X'], y=row['Y'], s=row.name)

        if (ellipse_outline or ellipse_fill) and color_labels is not None:
            for (label, group), color in zip(group_by, colors):
                x_mean, y_mean, width, height, angle = plot_util.build_ellipse(group['X'], group['Y'])
                ax.add_patch(mpl.patches.Ellipse(
                    (x_mean, y_mean),
                    width,
                    height,
                    angle=angle,
                    linewidth=2 if ellipse_outline else 0,
                    color=color,
                    fill=ellipse_fill,
                    alpha=0.2 + (0.3 if not show_points else 0) if ellipse_fill else 1
                ))

        if not supplementary_principal_coordinates.empty:
            active_legend = mlines.Line2D([], [], marker='.', linestyle='', color=GRAYS['dark'],
                                          markersize=14, label='Active rows')
            supp_legend = mlines.Line2D([], [], marker='*', linestyle='', color=GRAYS['dark'],
                                             markersize=14, label='Supplementary rows')
            ax.legend(handles=[active_legend, supp_legend])

        ax.set_title('Row principal coordinates')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax


    def correlation_circle(self, axes, column_correlations, supplementary_column_correlations,
                           explained_inertia, show_labels):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.axhline(y=0, linestyle='--', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)
        ax.axvline(x=0, linestyle='--', linewidth=1.2, color=GRAYS['dark'], alpha=0.6)

        # Plot the arrows and add text
        for _, row in column_correlations.iterrows():
            ax.annotate(
                row.name if show_labels else '',
                xy=(0, 0),
                xytext=(row[axes[0]], row[axes[1]]),
                arrowprops=dict(arrowstyle='<-', edgecolor='black')
            )

        if not supplementary_column_correlations.empty:
            for _, row in supplementary_column_correlations.iterrows():
                ax.annotate(
                    row.name if show_labels else '',
                    xy=(0, 0),
                    xytext=(row[axes[0]], row[axes[1]]),
                    arrowprops=dict(arrowstyle='<-', edgecolor='red')
                )
            # Add legend to distinguish active and supplementary columns
            active_legend = mpatches.Patch(color='black', label='Active columns')
            supp_legend = mpatches.Patch(color='red', label='Supplementary columns')
            plt.legend(handles=[active_legend, supp_legend])

        circle = plt.Circle((0, 0), radius=1, color=GRAYS['dark'], fill=False, lw=1.4)
        ax.add_patch(circle)

        ax.margins(0.01)
        ax.axis('equal')

        ax.set_title('Correlation circle')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax
