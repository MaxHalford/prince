import numpy as np
import pandas as pd

from .base import Base
from .plot.mpl.ca import MplCAPlotter
from .svd import SVD


class CA(Base):

    """Correspondance Analysis"""

    def __init__(self, dataframe, nbr_components=2, plotter='mpl'):
        super(CA, self).__init__(
            dataframe=dataframe,
            k=nbr_components,
            plotter=plotter
        )
        self._set_plotter(plotter)

        # Compute the relative frequency of each cell by dividing it by the total sum of all cells
        self.N = np.sum(self.X.values)
        self.P = self.X / self.N

        self._compute_svd()

    def _compute_svd(self):
        self.svd = SVD(X=self.standardized_residuals, k=self.k)

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplCAPlotter()
        }[plotter_name]

    @property
    def row_sums(self):
        """Compute row sums."""
        return self.P.sum(axis='columns')

    @property
    def column_sums(self):
        """Compute column sums."""
        return self.P.sum(axis='rows')

    @property
    def expected_frequencies(self):
        """Compute expected frequencies by performing a matrix multiplication between the rows sums
        and the column sums."""
        return pd.DataFrame(
            data=self.row_sums.values.reshape(-1, 1) @ self.column_sums.values.reshape(1, -1),
            index=self.P.index,
            columns=self.P.columns
        )

    @property
    def row_weights(self):
        return np.diag(1 / np.sqrt(self.row_sums))

    @property
    def column_weights(self):
        return np.diag(1 / np.sqrt(self.column_sums))

    @property
    def standardized_residuals(self):
        """The matrix of standardized residuals."""
        return self.row_weights @ (self.P - self.expected_frequencies).values @ self.column_weights

    @property
    def row_principal_components(self):
        """The row principal components."""
        return pd.DataFrame(
            data=self.row_weights @ self.svd.U @ np.diag(self.svd.s),
            index=self.P.index
        )

    @property
    def row_standard_components(self):
        """The row standard components by projecting X on it's right eigenvectors before dividing it
        by the eigenvalues."""
        principal_components = self.row_principal_components
        return principal_components.div(self.eigenvalues, axis='columns')

    @property
    def row_component_contributions(self):
        """The contribution of each row towards eacb principal component."""
        squared_row_pc = np.square(self.row_principal_components)
        squared_row_pc_scaled = squared_row_pc.multiply(self.row_sums, axis='rows')
        return squared_row_pc_scaled.div(self.eigenvalues, axis='columns')

    @property
    def row_cosine_similarities(self):
        """The cosine of the angle shaped by the row principal components and the eigenvectors."""
        squared_row_pc = np.square(self.row_principal_components)
        total_squares = squared_row_pc.sum(axis='columns')
        return squared_row_pc.div(total_squares, axis='rows')

    @property
    def row_profiles(self):
        return pd.DataFrame(
            data=np.diag(1 / self.row_sums) @ self.P,
            index=self.P.index,
            columns=self.P.columns
        )

    @property
    def column_principal_components(self):
        return pd.DataFrame(
            data=self.column_weights @ self.svd.V.T @ np.diag(self.svd.s),
            index=self.P.columns
        )

    @property
    def column_standard_components(self):
        return self.column_principal_components.div(self.eigenvalues, axis='rows')

    @property
    def column_component_contributions(self):
        """The contribution of each column towards each principal component."""
        squared_column_pc = np.square(self.column_principal_components)
        squared_column_pc_scaled = squared_column_pc.multiply(self.column_sums, axis='rows')
        return squared_column_pc_scaled.div(self.eigenvalues, axis='columns')

    @property
    def column_cosine_similarities(self):
        """The cosine of the angle shaped by the column principal components and the
        eigenvectors."""
        squared_column_pc = np.square(self.column_principal_components)
        total_squares = squared_column_pc.sum(axis='rows')
        return squared_column_pc.div(total_squares, axis='columns')

    @property
    def column_profiles(self):
        return pd.DataFrame(
            data=(np.diag(1 / self.column_sums) @ self.P.T).T,
            index=self.P.index,
            columns=self.P.columns
        )

    @property
    def total_inertia(self):
        """The total inertia is the sum of the relative differences between the observed and the
        expected frequencies."""
        expected = self.expected_frequencies
        return np.sum((np.square(self.P - expected) / expected).values)

    def plot_rows_columns(self, axes=(0, 1), show_row_points=True, show_row_labels=False,
                          show_column_points=True, show_column_labels=False):
        """Plot the row and column projections."""
        return self.plotter.row_column_projections(
            axes=axes,
            row_projections=self.row_principal_components,
            column_projections=self.column_principal_components,
            explained_inertia=self.explained_inertia,
            show_row_points=show_row_points,
            show_row_labels=show_row_labels,
            show_column_points=show_column_points,
            show_column_labels=show_column_labels
        )
