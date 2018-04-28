"""Multiple Correspondence Analysis (MCA)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils

from . import ca
from . import plot


class MCA(ca.CA):

    def fit(self, X, y=None):

        # Determine the number of columns in the initial matrix
        n_initial_columns = X.shape[1]

        # One-hot encode the dataset to retrieve an indicator matrix
        if isinstance(X, pd.DataFrame):
            X = pd.get_dummies(X.astype({col: 'category' for col in X.columns}))\
                  .astype(np.int8)
        else:
            X = pd.get_dummies(pd.DataFrame(X)).astype(np.int8)

        # Determine the number of columns in the indicator matrix
        n_new_columns = X.shape[1]

        # Apply correspondence analysis to the indicator matrix
        super().fit(X)

        # Compute the total inertia
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns

        return self

    def plot_principal_coordinates(self, ax=None, figsize=(7, 7), x_component=0, y_component=1,
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

            row_coords = self.row_principal_coordinates()

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

            col_coords = self.column_principal_coordinates()
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

            ax.legend(bbox_to_anchor=(1.04, 1), ncol=legend_n_cols)

        # Text
        ax.set_title('Row and column principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}%)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}%)'.format(y_component, 100 * ei[y_component]))

        return ax
