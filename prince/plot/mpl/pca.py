import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .. import util
from .. import PCAPlotter
from . import MplPlotter


class MplPCAPlotter(MplPlotter, PCAPlotter):

    """Matplotlib plotter for Principal Component Analysis"""

    def row_projections(self, axes, projections, explained_inertia, show_points, show_labels,
                        color_labels, ellipse_outline, ellipse_fill):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.axhline(y=0, linestyle='-', linewidth=1.2, color=self.colors['dark-gray'], alpha=0.6)
        ax.axvline(x=0, linestyle='-', linewidth=1.2, color=self.colors['dark-gray'], alpha=0.6)

        data = projections.iloc[:, axes].copy()
        data.columns = ('X', 'Y')

        if (color_labels is not None) and (show_points or show_labels or ellipse_outline or ellipse_fill):
            data['label'] = color_labels
            groups = data.groupby('label')
            labels = [label for label, _ in groups]

        if color_labels is not None:
            colors = self.qual_cmap(np.linspace(0, 1, len(groups)))
            cax = fig.add_axes((0.9, 0.1, 0.03, 0.8))
            bounds = list(range(len(groups)+1))
            norm = mpl.colors.BoundaryNorm(bounds, self.qual_cmap.N)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=self.qual_cmap, norm=norm, ticks=bounds)
            cbar.ax.set_yticklabels(labels)

        if show_points:
            if color_labels is not None:
                for (label, group), color in zip(groups, colors):
                    ax.scatter(group['X'], group['Y'], s=50, color=color, label=label)
            else:
                ax.scatter(data['X'], data['Y'], s=50, color=self.colors['blue'])

        if show_labels:
            ax.scatter(data['X'], data['Y'], alpha=0)
            if color_labels is not None:
                for (label, group), color in zip(groups, colors):
                    for _, row in group.iterrows():
                        ax.text(x=row['X'], y=row['Y'], s=row.name, color=color)
            else:
                for _, row in data.iterrows():
                    ax.text(x=row['X'], y=row['Y'], s=row.name)

        if (ellipse_outline or ellipse_fill) and color_labels is not None:
            for (label, group), color in zip(groups, colors):
                x_mean, y_mean, width, height, angle = util.build_ellipse(group['X'], group['Y'])
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

        ax.axis('equal')

        ax.set_title('Row projections')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax


    def correlation_circle(self, axes, variable_correlations, explained_inertia, show_labels):
        fig, ax = plt.subplots()

        ax.grid('on')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.axhline(y=0, linestyle='--', linewidth=1.2, color=self.colors['dark-gray'], alpha=0.6)
        ax.axvline(x=0, linestyle='--', linewidth=1.2, color=self.colors['dark-gray'], alpha=0.6)

        # Plot the arrows and add text
        for _, row in variable_correlations.iterrows():
            x = row[axes[0]]
            y = row[axes[1]]
            ax.annotate(
                row.name if show_labels else '',
                xy=(0, 0),
                xytext=(x, y),
                arrowprops={'arrowstyle': '<-'}
            )

        circle = plt.Circle((0, 0), radius=1, color=self.colors['dark-gray'], fill=False, lw=1.4)
        ax.add_patch(circle)

        ax.margins(0.01)
        ax.axis('equal')

        ax.set_title('Correlation circle')
        ax.set_xlabel('Component {} ({}%)'.format(axes[0], 100 * round(explained_inertia[axes[0]], 2)))
        ax.set_ylabel('Component {} ({}%)'.format(axes[1], 100 * round(explained_inertia[axes[1]], 2)))

        return fig, ax
