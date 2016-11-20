"""Correspondance Analysis"""
import numpy as np
import pandas as pd

from .base import Base
from .plot.mpl.ca import MplCAPlotter
from .svd import SVD


class CA(Base):

    """
    Args:
        dataframe (pandas.DataFrame): A contingency table.
        n_components (int): The number of principal components that have to be computed. The lower
            `n_components` is, the lesser time the CA will take to compute.
        plotter (str): The plotting backend used to build the charts. Can be any of: 'mpl'.
    """

    def __init__(self, dataframe, n_components=2, plotter='mpl'):
        super(CA, self).__init__(
            dataframe=dataframe,
            k=n_components,
            plotter=plotter
        )
        self._set_plotter(plotter_name=plotter)

        self._compute_svd()

    def _compute_svd(self):
        self.svd = SVD(X=self.standardized_residuals, k=self.n_components)

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplCAPlotter()
        }[plotter_name]

    @property
    def N(self):
        """The total number of occurences in `X`.

        Returns:
            int: The sum of each cell in `X`.
        """
        return np.sum(self.X.values)

    @property
    def P(self):
        """The relative contingency table.

        Obtained by dividing each cell in `X` by `N`.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `p`) of total sum 1.
        """
        return self.X / self.N


    @property
    def row_sums(self):
        """The sum of each row in `P`.

        Returns:
            pandas.Series: A series of length `n` representing the row values distribution.
        """
        return self.P.sum(axis='columns')

    @property
    def column_sums(self):
        """The sum of each column in `P`.

        Returns:
            pandas.Series: A series of length `p` representing the column values distribution.
        """
        return self.P.sum(axis='rows')

    @property
    def expected_frequencies(self):
        """The expected frequencies in case of total independence.

        Calculated by performing an outer product between the row sums and the column sums.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `p`) representing the row and column
            values distribution.
        """
        return pd.DataFrame(
            data=self.row_sums.values.reshape(-1, 1).dot(self.column_sums.values.reshape(1, -1)),
            index=self.P.index,
            columns=self.P.columns
        )

    @property
    def row_masses(self):
        """Diagonal matrix of row masses.

        Returns:
            numpy.ndarray: A two dimensional array of shape (`n`, `n`) where each diagonal value
            represents the weight of the matching row; the non-diagonal cells are equal to 0.
        """
        return np.diag(1 / np.sqrt(self.row_sums))

    @property
    def column_masses(self):
        """Diagonal matrix of column masses.

        Returns:
            numpy.ndarray: A two dimensional array of shape (`p`, `p`) where each diagonal value
            represents the weight of the matching column; the non-diagonal cells are equal to 0.
        """
        return np.diag(1 / np.sqrt(self.column_sums))

    @property
    def standardized_residuals(self):
        """The matrix of standardized residuals.

        Obtained by normalizing the differences between the relative contingency table and the
        expected table by the row and column masses. This is the matrix that is used to calculate
        the SVD for Correspondance Analysis.

        Returns:
            numpy.ndarray: A two dimensional array of shape (`n`, `p`).
        """
        residuals = (self.P - self.expected_frequencies).values
        return self.row_masses.dot(residuals).dot(self.column_masses)

    @property
    def row_standard_coordinates(self):
        """The row standard coordinates.

        The row standard coordinates are obtained by projecting the row masses on the left
        eigenvectors.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `k`) containing the row standard
            coordinates.
        """
        return pd.DataFrame(
            data=self.row_masses.dot(self.svd.U),
            index=self.P.index
        )

    @property
    def row_principal_coordinates(self):
        """The row principal coordinates.

        The row principal coordinates are obtained by multiplying the row standard coordinates by
        the singular values.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `k`) containing the row principal
            coordinates.
        """
        return pd.DataFrame(
            data=self.row_standard_coordinates.values.dot(np.diag(self.svd.s)),
            index=self.P.index
        )

    @property
    def row_component_contributions(self):
        """The row component contributions.

        Each row contribution towards each principal component is equivalent to the amount of
        inertia it contributes. This is calculated by dividing the scaled squared row coordinates by
        the eigenvalue associated to each principal component.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `k`) containing the row component
            contributions.
        """
        squared_coordinates = np.square(self.row_principal_coordinates)
        scaled_squared_coordinates = squared_coordinates.multiply(self.row_sums, axis='rows')
        return scaled_squared_coordinates.div(self.eigenvalues, axis='columns')

    @property
    def row_cosine_similarities(self):
        """The squared row cosine similarities.

        The row cosine similarities are obtained by calculating the cosine of the angle shaped by
        the row principal coordinates and the row principal components. This is calculated by
        squaring each row projection coordinate and dividing each squared coordinate by the sum of
        the squared coordinates, which results in a ratio comprised between 0 and 1 representing the
        squared cosine.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `k`) containing the squared row cosine
            similarities.
        """
        squared_coordinates = np.square(self.row_principal_coordinates)
        total_squares = squared_coordinates.sum(axis='columns')
        return squared_coordinates.div(total_squares, axis='rows')

    @property
    def row_profiles(self):
        """The row profiles.

        The row profiles are obtained by dividing each cell in the relative contingency table by the
        sum of the row it belongs to.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `p`) containing the row profiles.
        """
        return pd.DataFrame(
            data=np.diag(1 / self.row_sums).dot(self.P),
            index=self.P.index,
            columns=self.P.columns
        )

    @property
    def column_standard_coordinates(self):
        """The column standard coordinates.

        The column standard coordinates are obtained by projecting the column masses on the right
        eigenvectors.

        Returns:
            pandas.DataFrame: A dataframe of shape (`p`, `k`) containing the column standard
            coordinates.
        """
        return pd.DataFrame(
            data=self.column_masses.dot(self.svd.V.T),
            index=self.P.columns
        )

    @property
    def column_principal_coordinates(self):
        """The column principal coordinates.

        The column principal coordinates are obtained by multiplying the column standard coordinates
        by the singular values.

        Returns:
            pandas.DataFrame: A dataframe of shape (`p`, `k`) containing the column principal
            coordinates.
        """
        return pd.DataFrame(
            data=self.column_standard_coordinates.values.dot(np.diag(self.svd.s)),
            index=self.P.columns
        )

    @property
    def column_component_contributions(self):
        """The column component contributions.

        Each column contribution towards each principal component is equivalent to the amount of
        inertia it contributes. This is calculated by dividing the scaled squared column coordinates
        by the eigenvalue associated to each principal component.

        Returns:
            pandas.DataFrame: A dataframe of shape (`p`, `k`) containing the column component
            contributions.
        """
        squared_coordinates = np.square(self.column_principal_coordinates)
        scaled_squared_coordinates = squared_coordinates.multiply(self.column_sums, axis='rows')
        return scaled_squared_coordinates.div(self.eigenvalues, axis='columns')

    @property
    def column_cosine_similarities(self):
        """The squared column cosine similarities.

        The column cosine similarities are obtained by calculating the cosine of the angle shaped by
        the column principal coordinates and the column principal components. This is calculated by
        squaring each column projection coordinate and dividing each squared coordinate by the sum
        of the squared coordinates, which results in a ratio comprised between 0 and 1 representing
        the squared cosine.

        Returns:
            pandas.DataFrame: A dataframe of shape (`p`, `k`) containing the squared row cosine
            similarities.
        """
        squared_column_pc = np.square(self.column_principal_coordinates)
        total_squares = squared_column_pc.sum(axis='rows')
        return squared_column_pc.div(total_squares, axis='columns')

    @property
    def column_profiles(self):
        """The column profiles.

        The column profiles are obtained by dividing each cell in the relative contingency table by
        the sum of the column it belongs to.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `p`) containing the column profiles.
        """
        return pd.DataFrame(
            data=(np.diag(1 / self.column_sums).dot(self.P.T).T),
            index=self.P.index,
            columns=self.P.columns
        )

    @property
    def total_inertia(self):
        """The total inertia.

        Obtained by summing up the squared relative differences between the relative contingency
        table and the expected frequencies.

        Returns:
            float: The total inertia.
        """
        expected = self.expected_frequencies
        return np.sum((np.square(self.P - expected) / expected).values)

    def plot_rows_columns(self, axes=(0, 1), show_row_points=True, show_row_labels=False,
                          show_column_points=True, show_column_labels=False):
        """Plot the row and column principal coordinates.

        Args:
            axes (List(int)): A list of length two indicating which row principal coordinates to
                display.
            show_row_points (bool): Whether or not to show a point for each row principal
                coordinate.
            show_row_labels (bool): Whether or not to show the name of each row principal
                coordinate.
            show_column_points (bool): Whether or not to show a point for each column principal
                coordinate.
            show_column_labels (bool): Whether or not to show the name of each column principal
                coordinate.
        """
        return self.plotter.row_column_principal_coordinates(
            axes=axes,
            row_principal_coordinates=self.row_principal_coordinates,
            column_principal_coordinates=self.column_principal_coordinates,
            explained_inertia=self.explained_inertia,
            show_row_points=show_row_points,
            show_row_labels=show_row_labels,
            show_column_points=show_column_points,
            show_column_labels=show_column_labels
        )
