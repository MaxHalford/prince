"""Multiple Correspondence Analysis (MCA)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils

from . import ca
from . import one_hot
from . import plot


class MCA(ca.CA):

    def fit(self, X, y=None):

        utils.check_array(X, dtype=[str, np.number])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_initial_columns = X.shape[1]

        # One-hot encode the data
        self.one_hot_ = one_hot.OneHotEncoder().fit(X)

        # Apply CA to the indicator matrix
        super().fit(self.one_hot_.transform(X))

        # Compute the total inertia
        n_new_columns = len(self.one_hot_.column_names_)
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns

        return self

    def row_coordinates(self, X):
        return super().row_coordinates(self.one_hot_.transform(X))

    def column_coordinates(self, X):
        return super().column_coordinates(self.one_hot_.transform(X))

    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 's_')
        utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(X)

    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                                   show_row_points=True, row_points_size=10, show_row_labels=False,
                                   show_column_points=True, column_points_size=30,
                                   show_column_labels=False, legend_n_cols=1):
        """Plot row and column principal coordinates.

        Args:
            ax (matplotlib.Axis): A fresh one will be created and returned if not provided.
            figsize ((float, float)): The desired figure size if `ax` is not provided.
            x_component (int): Number of the component used for the x-axis.
            y_component (int): Number of the component used for the y-axis.
            show_row_points (bool): Whether to show row principal components or not.
            row_points_size (float): Row principal components point size.
            show_row_labels (bool): Whether to show row labels or not.
            show_column_points (bool): Whether to show column principal components or not.
            column_points_size (float): Column principal components point size.
            show_column_labels (bool): Whether to show column labels or not.
            legend_n_cols (int): Number of columns used for the legend.

        Returns:
            matplotlib.Axis
        """

        utils.validation.check_is_fitted(self, 'total_inertia_')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = plot.stylize_axis(ax)

        # Plot row principal coordinates
        if show_row_points or show_row_labels:

            row_coords = self.row_coordinates(X)

            if show_row_points:
                ax.scatter(
                    row_coords.iloc[:, x_component],
                    row_coords.iloc[:, y_component],
                    s=row_points_size,
                    label=None,
                    color=plot.GRAY['dark'],
                    alpha=0.6
                )

            if show_row_labels:
                for _, row in row_coords.iterrows():
                    ax.annotate(row.name, (row[x_component], row[y_component]))

        # Plot column principal coordinates
        if show_column_points or show_column_labels:

            col_coords = self.column_coordinates(X)
            x = col_coords[x_component]
            y = col_coords[y_component]

            prefixes = col_coords.index.str.split('_').map(lambda x: x[0])

            for prefix in prefixes.unique():
                mask = prefixes == prefix

                if show_column_points:
                    ax.scatter(x[mask], y[mask], s=column_points_size, label=prefix)

                if show_column_labels:
                    for i, label in enumerate(col_coords[mask].index):
                        ax.annotate(label, (x[mask][i], y[mask][i]))

            ax.legend(ncol=legend_n_cols)

        # Text
        ax.set_title('Row and column principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax
