"""Principal Component Analysis"""
import numpy as np
import pandas as pd

from . import util
from .base import Base
from .plot.mpl.pca import MplPCAPlotter
from .svd import SVD


class PCA(Base):

    """
    Args:
        dataframe (pandas.DataFrame): The columns containing categorical data will be removed from
            the dataframe provided by the user; they will be stored in a separate dataframe at
            initialization.
        n_components (int): The number of principal components that have to be computed. The lower
            `n_components` is, the lesser time the PCA will take to compute.
        scaled (bool): Whether or not to rescale each variable by subtracting it's mean to it and
            then dividing it by it's standard deviation. This is advised when variables are not of
            the same order of magnitude.
        supplementary_rows (List[int/str]): A list of rows that won't be used to compute the PCA.
            These rows can however be displayed together with the active rows on a row
            principal coordinates chart.
        supplementary_columns (List[str]): A list of columns that won't be used to compute the PCA.
            The columns can however be displayed together with the active columns on a column
            correlation chart.
        plotter (str): The plotting backend used to build the charts. Can be any of: 'mpl'.
    """

    def __init__(self, dataframe, n_components=2, scaled=True, supplementary_rows=None,
                 supplementary_columns=None, plotter='mpl'):

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('dataframe muse be a pandas.DataFrame')

        self.categorical_columns = pd.DataFrame()
        self.supplementary_columns = pd.DataFrame()
        self.supplementary_rows = pd.DataFrame()

        self._filter(
            dataframe=dataframe,
            supplementary_row_names=supplementary_rows if supplementary_rows else [],
            supplementary_column_names=supplementary_columns if supplementary_columns else []
        )
        super(PCA, self).__init__(
            dataframe=dataframe,
            k=n_components,
            plotter=plotter
        )
        self._set_plotter(plotter_name=plotter)

        self.scaled = scaled
        if self.scaled:
            mean = self.X.mean()
            std = self.X.std()
            self.X = (self.X - mean) / std
            if not self.supplementary_rows.empty:
                self.supplementary_rows -= mean
                self.supplementary_rows /= std

        self._compute_svd()

    def _compute_svd(self):
        self.svd = SVD(X=self.X.values, k=self.n_components)

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplPCAPlotter()
        }[plotter_name]

    def _filter(self, dataframe, supplementary_row_names, supplementary_column_names):

        # Extract the categorical columns
        self.categorical_columns = dataframe.select_dtypes(exclude=[np.number])

        # Extract the supplementary rows
        self.supplementary_rows = dataframe.loc[supplementary_row_names].copy()
        self.supplementary_rows.drop(self.categorical_columns.columns, axis=1, inplace=True)

        # Extract the supplementary columns
        self.supplementary_columns = dataframe[supplementary_column_names].copy()
        self.supplementary_columns.drop(supplementary_row_names, axis=0, inplace=True)

        # Remove the categorical column and the supplementary columns and rows from the dataframe
        dataframe.drop(supplementary_row_names, axis=0, inplace=True)
        dataframe.drop(supplementary_column_names, axis=1, inplace=True)
        dataframe.drop(self.categorical_columns.columns, axis=1, inplace=True)

    @property
    def n_supplementary_rows(self):
        """The number of supplementary rows."""
        return self.supplementary_rows.shape[0]

    @property
    def n_supplementary_columns(self):
        """The number of supplementary columns."""
        return self.supplementary_columns.shape[1]

    @property
    def row_principal_coordinates(self):
        """The row principal coordinates.

        The row principal coordinates are obtained by projecting `X` on it's right eigenvectors.
        This is done by calculating the dot product between `X` and `X`'s right eigenvectors.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `k`) containing the row principal
            coordinates.
        """
        return pd.DataFrame(data=self.X.dot(self.svd.V.T), index=self.X.index)

    @property
    def supplementary_row_principal_coordinates(self):
        """The supplementary row principal coordinates.

        The row principal coordinates are obtained by projecting the supplementary rows on the right
        eigenvectors of `X`.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n_supplementary_rows`, `k`) containing the
            supplementary row principal coordinates.
        """
        return pd.DataFrame(
            data=self.supplementary_rows.dot(self.svd.V.T),
            index=self.supplementary_rows.index
        )

    @property
    def row_standard_coordinates(self):
        """The row standard coordinates.

        The row standard coordinates are obtained by scaling/dividing each row projection by it's
        associated eigenvalue.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `k`) containing the row standard
            coordinates.
        """
        return self.row_principal_coordinates.div(self.eigenvalues, axis='columns')

    @property
    def supplementary_row_standard_components(self):
        """The supplementary row standard coordinates.

        The supplementary row standard coordinates are obtained by scaling/dividing each
        supplementary row projection by it's associated eigenvalue.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n_supplementary_rows`, `k`) containing the
            supplementary row standard coordinates.
        """
        return self.supplementary_row_principal_coordinates.div(self.eigenvalues, axis='columns')

    @property
    def row_component_contributions(self):
        """The row component contributions.

        Each row contribution towards each principal component is equivalent to the amount of
        inertia it contributes. This is calculated by dividing the squared row coordinates by the
        eigenvalue associated to each principal component.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n`, `k`) containing the row component
            contributions.
        """
        squared_coordinates = np.square(self.row_principal_coordinates)
        return squared_coordinates.div(self.eigenvalues, axis='columns')

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
    def supplementary_row_cosine_similarities(self):
        """The supplementary squared row cosine similarities.

        The supplementary row cosine similarities are obtained by calculating the cosine of the
        angle shaped by the supplementary row principal coordinates and the supplementary row
        principal components.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n_supplementary_rows`, `k`) containing the
            squared supplementary row cosine similarities.
        """
        squared_coordinates = np.square(self.supplementary_row_principal_coordinates)
        total_squares = squared_coordinates.sum(axis='columns')
        return squared_coordinates.div(total_squares, axis='rows')

    @property
    def column_correlations(self):
        """The column correlations with each principal component.

        Returns:
            pandas.DataFrame: A dataframe of shape (`p`, `k`) containing the Pearson
            correlations between the columns and the principal components.
        """
        row_pc = self.row_principal_coordinates
        return pd.DataFrame(
            data=([col.corr(pc) for _, pc in row_pc.iteritems()] for _, col in self.X.iteritems()),
            columns=row_pc.columns,
            index=self.X.columns
        )

    @property
    def supplementary_column_correlations(self):
        """The supplementary column correlations with each principal component.

        Returns:
            pandas.DataFrame: A dataframe of shape (`n_supplementary_columns`, `k`) containing
            the Pearson correlations between the supplementary columns and the principal components.
        """
        row_pc = self.row_principal_coordinates
        return pd.DataFrame(
            data=(
                [
                    col.corr(pc) if col.dtype in ('int64', 'float64')
                                 else util.correlation_ratio(col, pc)
                    for _, pc in row_pc.iteritems()
                ]
                for _, col in self.supplementary_columns.iteritems()
            ),
            columns=row_pc.columns,
            index=self.supplementary_columns.columns
        )

    @property
    def total_inertia(self):
        """The total inertia.

        Obtained by summing up the variance of each column.

        Returns:
            float: The total inertia.
        """
        return np.sum(np.square(self.X.values))

    def plot_rows(self, axes=(0, 1), show_points=True, show_labels=False, color_by=None,
                  ellipse_outline=False, ellipse_fill=False):
        """Plot the row principal coordinates.

        Args:
            axes (List(int)): A list of length two indicating which row principal coordinates to
                display.
            show_points (bool): Whether or not to show a point for each row principal coordinate.
            show_labels (bool): Whether or not to show the name of each row principal coordinate.
            color_by (str): Indicates according to which categorical variable the information should
                be colored by.
            ellipse_outline (bool): Whether or not to display an ellipse outline around each class
                if `color_by` has been set.
            ellipse_fill (bool): Whether or not to display a filled ellipse around each class if
                `color_by` has been set.
        """

        # Get color labels
        if color_by is None:
            color_labels = None
        elif color_by not in self.categorical_columns.columns:
            raise ValueError("'{}' is not a categorial column".format(color_by))
        else:
            color_labels = self.categorical_columns[color_by]

        return self.plotter.row_principal_coordinates(
            axes=axes,
            principal_coordinates=self.row_principal_coordinates,
            supplementary_principal_coordinates=self.supplementary_row_principal_coordinates,
            explained_inertia=self.explained_inertia,
            show_points=show_points,
            show_labels=show_labels,
            color_labels=color_labels,
            ellipse_outline=ellipse_outline,
            ellipse_fill=ellipse_fill
        )

    def plot_correlation_circle(self, axes=(0, 1), show_labels=True):
        """Plot the Pearson correlations between the components and the original columns.

        Args:
            axes (List(int)): A list of length two indicating which row principal coordinates to
                display.
            show_labels (bool): Whether or not to show the name of each column.
        """
        return self.plotter.correlation_circle(
            axes=axes,
            column_correlations=self.column_correlations,
            supplementary_column_correlations=self.supplementary_column_correlations,
            explained_inertia=self.explained_inertia,
            show_labels=show_labels
        )
