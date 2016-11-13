import numpy as np
import pandas as pd

from . import util
from .base import Base
from .plot.mpl.pca import MplPCAPlotter
from .svd import SVD


class PCA(Base):

    """Principal Component Analysis"""

    def __init__(self, dataframe, nbr_components=2, scaled=True, supplementary_rows=None,
                 supplementary_columns=None, plotter='mpl'):

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('dataframe muse be a pandas.DataFrame')

        self.categorical_columns = pd.DataFrame()
        self.supplementary_columns = pd.DataFrame()
        self.supplementary_rows = pd.DataFrame()

        self.scaled = scaled
        self._set_plotter(plotter)

        self._filter(
            dataframe=dataframe,
            supplementary_row_names=supplementary_rows if supplementary_rows else [],
            supplementary_column_names=supplementary_columns if supplementary_columns else []
        )
        super(PCA, self).__init__(
            dataframe=dataframe,
            k=nbr_components,
            plotter=plotter
        )

        if self.scaled:
            mean = self.X.mean()
            std = self.X.std()
            self.X -= mean
            self.X /= std
            if not self.supplementary_rows.empty:
                self.supplementary_rows -= mean
                self.supplementary_rows /= std

        self._compute_svd()

    def _compute_svd(self):
        self.svd = SVD(X=self.X.values, k=self.k)

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplPCAPlotter()
        }[plotter_name]

    def _filter(self, dataframe, supplementary_row_names, supplementary_column_names):

        # The categorical columns are the ones whose values are not numerical
        categorical_column_names = [
            column
            for column in dataframe.columns
            if dataframe[column].dtype not in ('int64', 'float64')
        ]

        # The categorical and the supplementary columns can be dropped once extracted
        columns_to_drop = set(categorical_column_names + supplementary_column_names)

        # Extract the supplementary rows
        self.supplementary_rows = dataframe.loc[supplementary_row_names].copy()
        self.supplementary_rows.drop(columns_to_drop, axis=1, inplace=True)

        # Extract the supplementary columns
        self.supplementary_columns = dataframe[supplementary_column_names].copy()
        self.supplementary_columns.drop(supplementary_row_names, axis=0, inplace=True)

        # Extract the categorical columns
        self.categorical_columns = dataframe[categorical_column_names].copy()

        # Remove the categorical and the supplementary columns from the main dataframe
        dataframe.drop(supplementary_row_names, axis=0, inplace=True)
        dataframe.drop(columns_to_drop, axis=1, inplace=True)

    @property
    def row_principal_components(self):
        """A dataframe of shape (`n`, `k`) containing the row principal components obtained by
        projecting `X` on it's right eigenvectors."""
        return pd.DataFrame(data=self.X @ self.svd.V.T, index=self.X.index)

    @property
    def supplementary_row_principal_components(self):
        """A dataframe of shape (*, `k`) containing the supplementary row principal components."""
        return pd.DataFrame(
            data=self.supplementary_rows @ self.svd.V.T,
            index=self.supplementary_rows.index
        )

    @property
    def row_standard_components(self):
        """A dataframe of shape (`n`, `k`) containing the row principal components divided by their
        respective eigenvalues."""
        return self.row_principal_components.div(self.eigenvalues, axis='columns')

    @property
    def supplementary_row_standard_components(self):
        """A dataframe of shape (*, `k`) containing the supplementary row standard components."""
        return self.supplementary_row_principal_components.div(self.eigenvalues, axis='columns')

    @property
    def row_cosine_similarities(self):
        """A dataframe of shape (`n`, `k`) dataframe containing the cosine of the angle shaped by
        the row projections and the row principal components."""
        squared_row_pc = np.square(self.row_principal_components)
        total_squares = squared_row_pc.sum(axis='columns')
        return squared_row_pc.div(total_squares, axis='rows')

    @property
    def supplementary_row_cosine_similarities(self):
        """A dataframe of shape (*, `k`) dataframe containing the cosine similarities of the
        supplementary rows."""
        squared_row_pc = np.square(self.supplementary_row_principal_components)
        total_squares = squared_row_pc.sum(axis='columns')
        return squared_row_pc.div(total_squares, axis='rows')

    @property
    def row_component_contributions(self):
        """The contribution of each row towards each principal component."""
        squared_row_pc = np.square(self.row_principal_components)
        return squared_row_pc.div(self.eigenvalues, axis='columns')

    @property
    def column_correlations(self):
        """A dataframe of shape (`p`, `k`) dataframe containing the Pearson correlations between the
        active columns and the row principal components."""
        row_pc = self.row_principal_components
        return pd.DataFrame(
            data=([col.corr(pc) for _, pc in row_pc.iteritems()] for _, col in self.X.iteritems()),
            columns=row_pc.columns,
            index=self.X.columns
        )

    @property
    def supplementary_column_correlations(self):
        """A dataframe of shape (*, `k`) dataframe containing the Pearson correlations between the
        supplementary columns and the row principal components."""
        row_pc = self.row_principal_components
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
        """The total inertia obtained by summing up the variance of each column."""
        return np.sum(np.square(self.X.values))

    def plot_rows(self, axes=(0, 1), show_points=True, show_labels=False, color_by=None,
                  ellipse_outline=False, ellipse_fill=False):
        """Plot the row projections."""

        # Get color labels
        if color_by is None:
            color_labels = None
        elif color_by not in self.categorical_columns.columns:
            raise ValueError("'{}' is not a categorial column".format(color_by))
        else:
            color_labels = self.categorical_columns[color_by]

        return self.plotter.row_projections(
            axes=axes,
            projections=self.row_principal_components,
            supplementary_projections=self.supplementary_row_principal_components,
            explained_inertia=self.explained_inertia,
            show_points=show_points,
            show_labels=show_labels,
            color_labels=color_labels,
            ellipse_outline=ellipse_outline,
            ellipse_fill=ellipse_fill
        )

    def plot_correlation_circle(self, axes=(0, 1), show_labels=True):
        """Plot the Pearson correlations between the components and the original columns."""
        return self.plotter.correlation_circle(
            axes=axes,
            column_correlations=self.column_correlations,
            supplementary_column_correlations=self.supplementary_column_correlations,
            explained_inertia=self.explained_inertia,
            show_labels=show_labels
        )
