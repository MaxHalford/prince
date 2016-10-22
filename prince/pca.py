import numpy as np
import pandas as pd

from . import util
from .base import Base
from .plot.mpl.pca import MplPCAPlotter
from .svd import SVD


class PCA(Base):

    """Principal Component Analysis"""

    categorical_variables = pd.DataFrame()

    def __init__(self, dataframe, nbr_components=2, ignored_variable_names=(), scaled=True,
                 plotter='mpl'):

        self.ignored_variable_names = ignored_variable_names
        self.scaled = scaled

        util.verify_dataframe(dataframe)
        self._tidy(dataframe)

        super(PCA, self).__init__(dataframe=dataframe, k=nbr_components, plotter=plotter)

        self._set_plotter(plotter)

        self._preprocess(self.X)
        self._compute_svd()

    def _compute_svd(self):
        self.svd = SVD(X=self.X.values, k=self.k)

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplPCAPlotter()
        }[plotter_name]

    def _preprocess(self, X):
        """Center and rescale a dataframe."""
        X -= self.X.mean()
        if self.scaled:
            X /= self.X.std()

    def _tidy(self, dataframe):
        """Remove the ignored columns and stash the categorical columns."""
        for column in dataframe.columns:
            # Variable is ignored
            if column in self.ignored_variable_names:
                del dataframe[column]
            # Variable is categorical
            elif dataframe[column].dtype not in ('int64', 'float64'):
                self.categorical_variables[column] = dataframe[column]
                del dataframe[column]

    def project_rows(self, X, preprocess=True):
        """Project rows on a new subspace formed by a family of eigenvectors."""

        util.verify_dataframe(X)

        # Supplementary variables should be preprocessed the same way the active variables were
        if preprocess:
            X = X.copy(deep=True)
            self._preprocess(X)

        return X.values @ self.svd.V.T

    @property
    def row_principal_components(self):
        """A `n` by `k` dataframe containing the row principal components obtained by projecting X
        on it's right eigenvectors."""
        return pd.DataFrame(data=self.project_rows(self.X, False), index=self.X.index)

    @property
    def row_standard_components(self):
        """A `n` by `k` dataframe containing the row principal components obtained by projecting X
        on it's right eigenvectors before dividing them by their respective eigenvalues."""
        principal_components = self.row_principal_components
        return principal_components.div(self.eigenvalues, axis='columns')

    @property
    def row_component_contributions(self):
        """The contribution of each row towards each principal component."""
        squared_row_pc = np.square(self.row_principal_components)
        return squared_row_pc.div(self.eigenvalues, axis='columns')

    @property
    def row_cosine_similarities(self):
        """An `n` by `k` dataframe containing the cosine of the angle shaped by the row projections
        and the row principal components."""
        squared_row_pc = np.square(self.row_principal_components)
        total_squares = squared_row_pc.sum(axis='columns')
        return squared_row_pc.div(total_squares, axis='rows')

    @property
    def variable_correlations(self):
        """A `p` by `k` dataframe containing the Pearson correlations between the initial variables
        and the row principal components."""
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
        elif color_by not in self.categorical_variables:
            raise ValueError("Categorical variable '{}' can not be found".format(color_by))
        else:
            color_labels = self.categorical_variables[color_by]

        return self.plotter.row_projections(
            axes=axes,
            projections=self.row_principal_components,
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
