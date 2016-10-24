import pandas as pd

from . import util
from .ca import CA
from .plot.mpl.mca import MplMCAPlotter


class MCA(CA):

    """Multiple Correspondence Analysis

    Attributes:
        q (int): The number of columns in the initial dataframe; as opposed to `p` which is the
            number of columns in the indicator matrix of the initial dataframe.
    """

    numerical_variables = pd.DataFrame()

    def __init__(self, dataframe, nbr_components=2, ignored_variable_names=(), plotter='mpl',
                 use_benzecri_rates=False):

        util.verify_dataframe(dataframe)

        self.initial_dataframe = dataframe.copy(deep=True)
        self.use_benzecri_rates = use_benzecri_rates
        self.ignored_variable_names = ignored_variable_names

        self._tidy(dataframe)

        super(MCA, self).__init__(
            dataframe=pd.get_dummies(dataframe),
            nbr_components=nbr_components,
            plotter=plotter
        )

        # Keep the number of columns in the original dataset
        self.q = dataframe.shape[1]

    def _tidy(self, dataframe):
        """Remove the ignored columns and stash the numerical columns."""
        self.numerical_variables = pd.DataFrame()
        for column in dataframe.columns:
            # Variable is ignored
            if column in self.ignored_variable_names:
                del dataframe[column]
            # Variable is categorical
            elif dataframe[column].dtype in ('int64', 'float64'):
                self.numerical_variables[column] = dataframe[column]
                del dataframe[column]

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplMCAPlotter()
        }[plotter_name]

    @property
    def eigenvalues(self):
        """Compute the eigenvalues by squaring the singular values. If `use_benzecri_rates` is
        `True` Benzécri correction is applied to each eigenvalue."""
        eigenvalues = super(CA, self).eigenvalues
        if self.use_benzecri_rates:
            return util.calculate_benzecri_correction(eigenvalues)
        return eigenvalues

    @property
    def variable_correlations(self):
        """A `q` by `k` dataframe containing the correlation ratios between the initial variables
        and the row principal components. The correlations ratios are the inter-group variances
        divided by the sum of the inter-group and intra-group variance of the numerical values
        associated to each categorical variable.
        """
        return pd.DataFrame({
            variable.name: [
                util.correlation_ratio(variable.tolist(), principal_component)
                for _, principal_component in self.row_principal_components.iteritems()
            ]
            for _, variable in self.initial_dataframe.iteritems()
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
        elif color_by not in self.initial_dataframe:
            raise ValueError("Categorical variable '{}' can not be found".format(color_by))
        else:
            color_labels = self.initial_dataframe[color_by]

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
        """Plot the relationship square between the initial variables and the row principal
        components."""
        return self.plotter.relationship_square(
            axes=axes,
            variable_correlations=self.variable_correlations,
            explained_inertia=self.explained_inertia,
            show_labels=show_labels
        )
