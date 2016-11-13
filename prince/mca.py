import pandas as pd

from . import util
from .ca import CA
from .pca import PCA
from .plot.mpl.mca import MplMCAPlotter


class MCA(CA):

    """Multiple Correspondence Analysis

    Attributes:
        q (int): The number of columns in the initial dataframe; as opposed to `p` which is the
            number of columns in the indicator matrix of the initial dataframe.
    """

    def __init__(self, dataframe, nbr_components=2, supplementary_rows=None,
                 supplementary_columns=None, use_benzecri_rates=False, plotter='mpl'):

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('dataframe muse be a pandas.DataFrame')

        self.categorical_columns = pd.DataFrame()
        self.supplementary_columns = pd.DataFrame()
        self.supplementary_rows = pd.DataFrame()

        self.initial_dataframe = dataframe.copy(deep=True)
        self.use_benzecri_rates = use_benzecri_rates

        self._filter(
            dataframe=dataframe,
            supplementary_row_names=supplementary_rows if supplementary_rows else [],
            supplementary_column_names=supplementary_columns if supplementary_columns else []
        )
        self.q = dataframe.shape[1] # Number of columns in the original dataset

        super(MCA, self).__init__(
            dataframe=pd.get_dummies(dataframe),
            nbr_components=nbr_components,
            plotter=plotter
        )

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplMCAPlotter()
        }[plotter_name]

    def _filter(self, dataframe, supplementary_row_names, supplementary_column_names):

        # The categorical columns are the ones whose values are not numerical
        categorical_column_names = [
            column
            for column in dataframe.columns
            if dataframe[column].dtype not in ('int64', 'float64')
        ]

        # Extract the supplementary rows
        self.supplementary_rows = dataframe.loc[supplementary_row_names].copy()
        self.supplementary_rows.drop(supplementary_column_names, axis=1, inplace=True)

        # Extract the supplementary columns
        self.supplementary_columns = dataframe[supplementary_column_names].copy()
        self.supplementary_columns.drop(supplementary_row_names, axis=0, inplace=True)

        # Extract the categorical columns
        self.categorical_columns = dataframe[categorical_column_names].copy()

        # Remove the categorical and the supplementary columns from the main dataframe
        dataframe.drop(supplementary_row_names, axis=0, inplace=True)
        dataframe.drop(supplementary_column_names, axis=1, inplace=True)

    @property
    def eigenvalues(self):
        """Compute the eigenvalues by squaring the singular values. If `use_benzecri_rates` is
        `True` Benzécri correction is applied to each eigenvalue."""
        eigenvalues = super(CA, self).eigenvalues
        if self.use_benzecri_rates:
            return util.calculate_benzecri_correction(eigenvalues)
        return eigenvalues

    @property
    def column_correlations(self):
        """A `q` by `k` dataframe containing the correlation ratios between the initial columns
        and the row principal components. The correlations ratios are the inter-group variances
        divided by the sum of the inter-group and intra-group variance of the numerical values
        associated to each categorical column.
        """
        return pd.DataFrame({
            column.name: [
                util.correlation_ratio(column.tolist(), principal_component)
                for _, principal_component in self.row_principal_components.iteritems()
            ]
            for _, column in self.initial_dataframe.iteritems()
        }).T

    @property
    def total_inertia(self):
        return (self.p - self.q) / self.q

    def plot_rows(self, axes=(0, 1), show_points=True, show_labels=False, color_by=None,
                  ellipse_outline=False, ellipse_fill=False):
        """Plot the row projections."""

        # Get color labels
        if color_by is None:
            color_labels = None
        elif color_by not in self.initial_dataframe.columns:
            raise ValueError("Categorical column '{}' can not be found".format(color_by))
        else:
            color_labels = self.initial_dataframe[color_by]

        return self.plotter.row_projections(
            axes=axes,
            projections=self.row_principal_components,
            supplementary_projections=pd.DataFrame(columns=self.row_principal_components.columns), # To do
            explained_inertia=self.explained_inertia,
            show_points=show_points,
            show_labels=show_labels,
            color_labels=color_labels,
            ellipse_outline=ellipse_outline,
            ellipse_fill=ellipse_fill
        )

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

    def plot_relationship_square(self, axes=(0, 1), show_labels=True):
        """Plot the relationship square between the initial columns and the row principal
        components."""
        return self.plotter.relationship_square(
            axes=axes,
            column_correlations=self.column_correlations,
            explained_inertia=self.explained_inertia,
            show_labels=show_labels
        )
