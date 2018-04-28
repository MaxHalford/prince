"""Correspondence Analysis (CA)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import utils

from . import plot
from . import svd


class CA():

    def __init__(self, n_components=2, n_iter=10, copy=True, engine='auto'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.engine = engine

    def fit(self, X, y=None):

        # Check input
        utils.check_array(X)

        # Check all values are positive
        if np.any(X < 0):
            raise ValueError("All values in X should be positive")

        if isinstance(X, pd.DataFrame):
            self.row_label_ = X.index.name if X.index.name else 'Rows'
            self.row_names_ = X.index.tolist()
            self.col_label_ = X.columns.name if X.columns.name else 'Columns'
            self.col_names_ = X.columns.tolist()
            X = X.values
        else:
            self.row_label_ = 'Rows'
            self.row_names_ = list(range(X.shape[0]))
            self.col_label_ = 'Columns'
            self.col_names_ = list(range(X.shape[1]))

        if self.copy:
            X = np.copy(X)

        # Compute the correspondence matrix which contains the relative frequencies
        X = X / np.sum(X)

        # Compute row and column masses
        self.row_masses_ = pd.Series(X.sum(axis=1), index=self.row_names_)
        self.col_masses_ = pd.Series(X.sum(axis=0), index=self.col_names_)

        # Compute standardised residuals
        r = self.row_masses_
        c = self.col_masses_
        S = np.diag(r ** -0.5) @ (X - np.outer(r, c)) @ np.diag(c ** -0.5)

        # Compute SVD on the standardised residuals
        self.U_, self.s_, self.V_ = svd.compute_svd(S, self.n_components, self.n_iter, self.engine)

        # Compute total inertia
        self.total_inertia_ = (S @ S.T).trace()

        return self

    @property
    def eigenvalues_(self):
        """The eigenvalues associated with each principal component."""
        utils.validation.check_is_fitted(self, 's_')
        return np.square(self.s_).tolist()

    @property
    def explained_inertia_(self):
        """The percentage of explained inertia per principal component."""
        utils.validation.check_is_fitted(self, 'total_inertia_')
        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    def row_principal_coordinates(self):
        """The row principal coordinates."""
        utils.validation.check_is_fitted(self, 'U_')
        return pd.DataFrame(
            data=np.diag(self.row_masses_ ** -0.5) @ self.U_ @ np.diag(self.s_),
            index=self.row_names_
        )

    def column_principal_coordinates(self):
        """The column principal coordinates."""
        utils.validation.check_is_fitted(self, 'V_')
        return pd.DataFrame(
            data=np.diag(self.col_masses_ ** -0.5) @ self.V_.T @ np.diag(self.s_),
            index=self.col_names_
        )

    def plot_principal_coordinates(self, ax=None, figsize=(7, 7), x_component=0, y_component=1,
                                   show_row_labels=True, show_col_labels=True, **kwargs):
        """Plot the principal coordinates."""

        utils.validation.check_is_fitted(self, 'row_names_')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = plot.stylize_axis(ax)

        # Plot row principal coordinates
        row_coords = self.row_principal_coordinates()
        ax.scatter(
            row_coords[x_component],
            row_coords[y_component],
            **kwargs,
            label=self.row_label_
        )

        # Plot column principal coordinates
        col_coords = self.column_principal_coordinates()
        ax.scatter(
            col_coords[x_component],
            col_coords[y_component],
            **kwargs,
            label=self.col_label_
        )

        # Add row labels
        if show_row_labels:
            x = row_coords[x_component]
            y = row_coords[y_component]
            for i, label in enumerate(self.row_names_):
                ax.annotate(label, (x[i], y[i]))

        # Add column labels
        if show_col_labels:
            x = col_coords[x_component]
            y = col_coords[y_component]
            for i, label in enumerate(self.col_names_):
                ax.annotate(label, (x[i], y[i]))

        # Legend
        ax.legend()

        # Text
        ax.set_title('Principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}%)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}%)'.format(y_component, 100 * ei[y_component]))

        return ax
