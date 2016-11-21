"""Multiple Correspondence Analysis"""
import numpy as np
import pandas as pd

from . import util
from .ca import CA
from .plot.mpl.mca import MplMCAPlotter


class MCA(CA):

    """
    Args:
        dataframe (pandas.DataFrame): A dataframe where each column is a variable.
        n_components (int): The number of principal components that have to be computed. The lower
            `n_components` is, the lesser time the CA will take to compute.
        use_benzecri_rates (bool): Whether to use Benzecri rates to inflate the eigenvalues.
        plotter (str): The plotting backend used to build the charts. Can be any of: 'mpl'.
    """

    def __init__(self, dataframe, n_components=2, use_benzecri_rates=False, plotter='mpl'):

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('dataframe muse be a pandas.DataFrame')

        self.categorical_columns = pd.DataFrame()
        self.supplementary_columns = pd.DataFrame()
        self.supplementary_rows = pd.DataFrame()

        self.initial_dataframe = dataframe.copy(deep=True)
        self.use_benzecri_rates = use_benzecri_rates

        supplementary_rows = None # Todo
        supplementary_columns = None  # Todo

        self._filter(
            dataframe=dataframe,
            supplementary_row_names=supplementary_rows if supplementary_rows else [],
            supplementary_column_names=supplementary_columns if supplementary_columns else []
        )

        super(MCA, self).__init__(
            dataframe=pd.get_dummies(dataframe),
            n_components=n_components,
            plotter=plotter
        )

    def _set_plotter(self, plotter_name):
        self.plotter = {
            'mpl': MplMCAPlotter()
        }[plotter_name]

    def _filter(self, dataframe, supplementary_row_names, supplementary_column_names):

        # Extract the categorical columns
        self.categorical_columns = dataframe.select_dtypes(exclude=[np.number])

        # Extract the supplementary rows
        self.supplementary_rows = dataframe.loc[supplementary_row_names].copy()
        self.supplementary_rows.drop(supplementary_column_names, axis=1, inplace=True)

        # Extract the supplementary columns
        self.supplementary_columns = dataframe[supplementary_column_names].copy()
        self.supplementary_columns.drop(supplementary_row_names, axis=0, inplace=True)

        # Remove the the supplementary columns and rows from the dataframe
        dataframe.drop(supplementary_row_names, axis=0, inplace=True)
        dataframe.drop(supplementary_column_names, axis=1, inplace=True)

    @property
    def q(self):
        """The number of columns in the initial dataframe

        As opposed to `p` which is the number of columns in the indicator matrix of the initial
        dataframe.
        """
        return self.initial_dataframe.shape[1]

    @property
    def eigenvalues(self):
        """The eigenvalues associated to each principal component.

        The eigenvalues are obtained by squaring the singular values obtained from a SVD. If
        `use_benzecri_rates` is `True` then Benzécri correction is applied to each eigenvalue.

        Returns:
            List[float]: The eigenvalues ordered increasingly.
        """
        eigenvalues = super(CA, self).eigenvalues
        if self.use_benzecri_rates:
            return util.calculate_benzecri_correction(eigenvalues)
        return eigenvalues

    @property
    def column_correlations(self):
        """The column correlation ratios with each principal component.

        The correlations ratios are the inter-group variances divided by the sum of the inter-group
        and intra-group variance of the numerical values associated to each categorical column.

        Returns:
            pandas.DataFrame: A dataframe of shape (`q`, `k`) containing the Pearson
            correlations between the columns and the principal components.
        """
        return pd.DataFrame({
            column.name: [
                util.correlation_ratio(column.tolist(), principal_component)
                for _, principal_component in self.row_principal_coordinates.iteritems()
            ]
            for _, column in self.initial_dataframe.iteritems()
        }).T

    @property
    def total_inertia(self):
        """The total inertia."""
        return (self.n_columns - self.q) / self.q

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
        elif color_by not in self.initial_dataframe.columns:
            raise ValueError("Categorical column '{}' can not be found".format(color_by))
        else:
            color_labels = self.initial_dataframe[color_by]

        return self.plotter.row_principal_coordinates(
            axes=axes,
            principal_coordinates=self.row_principal_coordinates,
            supplementary_principal_coordinates=pd.DataFrame(
                columns=self.row_principal_coordinates.columns
            ), # To do
            explained_inertia=self.explained_inertia,
            show_points=show_points,
            show_labels=show_labels,
            color_labels=color_labels,
            ellipse_outline=ellipse_outline,
            ellipse_fill=ellipse_fill
        )

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

    def plot_relationship_square(self, axes=(0, 1), show_labels=True):
        """Plot the relationship square between the initial columns and the row principal
        components.

        Args:
            axes (List(int)): A list of length two indicating which row principal components to
                display.
            show_labels (bool): Whether or not to show the name of each column.
        """
        return self.plotter.relationship_square(
            axes=axes,
            column_correlations=self.column_correlations,
            explained_inertia=self.explained_inertia,
            show_labels=show_labels
        )
