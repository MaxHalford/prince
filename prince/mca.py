"""Multiple Correspondence Analysis (MCA)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.preprocessing import OneHotEncoder

from . import ca
from . import plot


class MCA(ca.CA):

    def fit(self, X, y=None, K=None):
        """Fit the MCA for the dataframe X.

        The MCA is computed on the indicator matrix (i.e. `X.get_dummies()`). If some of the columns are already
        in indicator matrix format, you'll want to pass in `K` as the number of "real" variables that it represents.
        (That's used for correcting the inertia linked to each dimension.)
        """

        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # K is the number of actual variables, to apply the Benzécri correction
        if K is None:
            self.K = X.shape[1]
        elif X.shape[1] < K:
            raise ValueError(f"K ({K}) can't be higher than number of columns ({X.shape[1]})")
        else:
            self.K = K

        # One-hot encode the data
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.enc.fit(X)
        one_hot = self.enc.transform(X)

        # We need the number of columns to apply the Greenacre correction
        self.J = one_hot.shape[1]

        # Apply CA to the indicator matrix
        super().fit(one_hot)

        self.total_inertia_ = np.sum(self.eigenvalues_)
        return self

    @property
    def eigenvalues_(self):
        """The eigenvalues associated with each principal component.

        This applies the Benzécri correction for MCA which corrects for the inflated dimensionality
        related to the extra columns of the indicator matrix.
        """
        self._check_is_fitted()

        K = self.K

        return np.array([
            (K / (K - 1.) * (s - 1. / K)) ** 2
            if s > 1. / K else 0
            for s in np.square(self.s_)
        ])


    @property
    def explained_inertia_(self):
        """The percentage of explained inertia per principal component.
        
        This applies the Greenacre correction to compensate for overestimation of 
        contribution.
        """
        self._check_is_fitted()
        K = self.K
        J = self.J

        # Average inertia on the diagonal of the Burt Matrix (JxJ)
        # s_ are the eigenvalues of the residials matrix. Square to obtain the eigenvalues of the Indicator matrix (IxJ), 
        # and square again for the eigenvalues of the Burt Matrix (JxJ)
        Theta = (K / (K - 1.)) * (np.sum(np.square(self.s_)**2) - (J-K)/(K**2))

        return self.eigenvalues_ / Theta

    def row_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        return super().row_coordinates(self.enc.transform(X))

    def column_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return super().column_coordinates(self.enc.transform(X))

    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        self._check_is_fitted()
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(X)

    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                         show_row_points=True, row_points_size=10,
                         row_points_alpha=0.6, show_row_labels=False,
                         show_column_points=True, column_points_size=30, show_column_labels=False,
                         legend_n_cols=1):
        """Plot row and column principal coordinates.

        Parameters:
            ax (matplotlib.Axis): A fresh one will be created and returned if not provided.
            figsize ((float, float)): The desired figure size if `ax` is not provided.
            x_component (int): Number of the component used for the x-axis.
            y_component (int): Number of the component used for the y-axis.
            show_row_points (bool): Whether to show row principal components or not.
            row_points_size (float): Row principal components point size.
            row_points_alpha (float): Alpha for the row principal component.
            show_row_labels (bool): Whether to show row labels or not.
            show_column_points (bool): Whether to show column principal components or not.
            column_points_size (float): Column principal components point size.
            show_column_labels (bool): Whether to show column labels or not.
            legend_n_cols (int): Number of columns used for the legend.

        Returns:
            matplotlib.Axis
        """

        self._check_is_fitted()

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
                    alpha=row_points_alpha
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
