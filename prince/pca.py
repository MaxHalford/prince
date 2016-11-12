import numpy as np
import pandas as pd

from .base import Base
from .plot.mpl.pca import MplPCAPlotter
from .svd import SVD


class PCA(Base):

    """Principal Component Analysis"""

    categorical_variables = pd.DataFrame()
    supplementary_variables = pd.DataFrame()
    supplementary_rows = pd.DataFrame()

    def __init__(self, dataframe, nbr_components=2, scaled=True, supplementary_rows=None,
                 supplementary_variables=None, plotter='mpl'):

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('dataframe muse be a pandas.DataFrame')

        self.scaled = scaled
        self._set_plotter(plotter)

        self._filter(
            dataframe=dataframe,
            supplementary_rows_names=supplementary_rows if supplementary_rows else [],
            supplementary_variable_names=supplementary_variables if supplementary_variables else []
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

    def _filter(self, dataframe, supplementary_rows_names, supplementary_variable_names):

        # The categorical variables are the ones whose values are not numerical
        categorical_variable_names = [
            column
            for column in dataframe.columns
            if dataframe[column].dtype not in ('int64', 'float64')
        ]

        # The categorical and the supplementary variables can be dropped once extracted
        columns_to_drop = set(categorical_variable_names + supplementary_variable_names)

        # Extract the supplementary rows
        self.supplementary_rows = dataframe.loc[supplementary_rows_names].copy()
        self.supplementary_rows.drop(columns_to_drop, axis=1, inplace=True)

        # Extract the supplementary variables
        self.supplementary_variables = dataframe[supplementary_variable_names].copy()

        # Extract the categorical variables
        self.categorical_variables = dataframe[categorical_variable_names].copy()

        # Remove the categorical and the supplementary variables from the main dataframe
        dataframe.drop(supplementary_rows_names, axis=0, inplace=True)
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
    def variable_correlations(self):
        """A dataframe of shape (`p`, `k`) dataframe containing the Pearson correlations between the
        initial variables and the row principal components."""
        row_pc = self.row_principal_components
        return pd.DataFrame(
            data=([col.corr(pc) for _, pc in row_pc.iteritems()] for _, col in self.X.iteritems()),
            columns=row_pc.columns,
            index=self.X.columns
        )

    @property
    def total_inertia(self):
        """The total inertia obtained by summing up the variance of each variable."""
        return np.sum(np.square(self.X.values))

    def plot_rows(self, axes=(0, 1), show_points=True, show_labels=False, color_by=None,
                  ellipse_outline=False, ellipse_fill=False):
        """Plot the row projections."""

        # Get color labels
        if color_by is None:
            color_labels = None
        elif color_by not in self.categorical_variables.columns:
            raise ValueError("Categorical variable '{}' can not be found".format(color_by))
        else:
            color_labels = self.categorical_variables[color_by]

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
        """Plot the Pearson correlations between the components and the original variables."""
        return self.plotter.correlation_circle(
            axes=axes,
            variable_correlations=self.variable_correlations,
            explained_inertia=self.explained_inertia,
            show_labels=show_labels
        )
