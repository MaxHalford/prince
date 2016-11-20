class Plotter():

    def inertia(self, explained_inertia):
        """Generate a Scree diagram displaying the inertia of each component.

        Args:
            explained_inertia (list(float))
        """
        raise NotImplementedError

    def cumulative_inertia(self, cumulative_explained_inertia, threshold):
        """Generate a Scree diagram displaying the cumulative inertia of each component.

        Args:
            cumulative_explained_inertia (list(float))
            threshold (float): The thresold at which a vertical line should be plotted.
        """
        raise NotImplementedError


class PCAPlotter():

    def row_principal_coordinates(self, axes, principal_coordinates,
                                  supplementary_principal_coordinates, explained_inertia,
                                  show_points, show_labels, color_labels, ellipse_outline,
                                  ellipse_fill):
        """Generate a plot displaying the row principal coordinates.

        Args:
            axes (list(int)): A list containing two distinct elements, each one indicating the index
                of a principal coordinates to plot.
            principal_coordinates (pandas.DataFrame)
            supplementary_principal_coordinates (pandas.DataFrame)
            explained_inertia (list(float))
            show_points (bool): Whether or not to display points.
            show_labels (bool): Whether or not to display labels above each point.
            color_labels (pandas.Series): A series containing ordinal values according to which each
                point and label will be colored.
            ellipse_outline (bool): Whether or not to draw the outline of an ellipse around each
                group defined with `color_labels`.
            ellipse_outline (bool): Whether or not to draw the filling of an ellipse around each
                group defined with `color_labels`.
        """
        raise NotImplementedError

    def correlation_circle(self, axes, column_correlations, supplementary_column_correlations,
                           explained_inertia, show_labels):
        """Generate a plot displaying the columns/component Pearson correlations.

        Args:
            axes (list(int)): A list containing two distinct elements, each one indicating the index
                of a principal coordinates to plot.
            column_correlations (pandas.DataFrame): A dataframe containing the correlations
                between each initial column and each row principal component.
            supplementary_column_correlations (pandas.DataFrame)
            explained_inertia (list(float))
            show_labels (bool): Indicates whether or not to display labels above each point.
        """
        raise NotImplementedError


class CAPlotter():

    def row_column_principal_coordinates(self, axes, row_principal_coordinates,
                                         column_principal_coordinates, explained_inertia,
                                         show_row_points, show_row_labels, show_column_points,
                                         show_column_labels):
        """Generate a plot displaying the row and column principal coordinates simultaneously.

        Args:
            axes (list(int)): A list containing two distinct elements, each one indicating the index
                of a principal coordinate to plot.
            row_principal_coordinates (pandas.DataFrame)
            column_principal_coordinates (pandas.DataFrame)
            explained_inertia (list(float))
            show_row_points (bool): Indicates whether or not to display points for the row
                principal_coordinates.
            show_row_labels (bool): Indicates whether or not to display labels above each row
                projection point.
            show_row_points (bool): Indicates whether or not to display points for the column
                principal_coordinates.
            show_row_labels (bool): Indicates whether or not to display labels above each column
                projection point.
        """
        raise NotImplementedError


class MCAPlotter():

    def row_column_principal_coordinates(self, axes, row_principal_coordinates,
                                         column_principal_coordinates, explained_inertia,
                                         show_row_points, show_row_labels, show_column_points,
                                         show_column_labels):
        """Generate a plot displaying the row and column principal coordinates simultaneously.

        Args:
            axes (list(int)): A list containing two distinct elements, each one indicating the index
                of a principal coordinate to plot.
            row_principal_coordinates (pandas.DataFrame)
            column_principal_coordinates (pandas.DataFrame)
            explained_inertia (list(float))
            show_row_points (bool): Indicates whether or not to display points for the row
                principal coordinates.
            show_row_labels (bool): Indicates whether or not to display labels above each row
                projection point.
            show_row_points (bool): Indicates whether or not to display points for the column
                principal coordinates.
            show_row_labels (bool): Indicates whether or not to display labels above each column
                projection point.
        """
        raise NotImplementedError

    def relationship_square(self, axes, column_correlations, explained_inertia, show_labels):
        """Generate a plit displaying the correlations ratios between the initial columns and the
        row principal components.

        Args:
            axes (list(int)): A list containing two distinct elements, each one indicating the index
                of a principal component to plot.
            column_correlations (pandas.DataFrame): A dataframe containing the correlations
                between each initial column and each row principal component.
            explained_inertia (list(float))
            show_labels (bool): Indicates whether or not to display labels above each point.
        """
        raise NotImplementedError

