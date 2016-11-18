import numpy as np
import pandas as pd


class Base():

    """Base contains the common operations performed during any factor analysis."""

    def __init__(self, dataframe, k, plotter):

        if plotter not in 'mpl':
            raise ValueError('Unrecognized plotting backend; choose from: mpl')

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('dataframe muse be a pandas.DataFrame')

        self.plotter = None
        self.svd = None

        self.__x = dataframe
        self.__n, self.__p = dataframe.shape
        # Determine the number of components computed during SVD
        self.__k = self.__p if k == -1 else min(k, self.__p)

    def _compute_svd(self):
        raise NotImplementedError

    def _set_plotter(self, plotter_name):
        raise NotImplementedError

    @property
    def X(self):
        """The dataset that is used to perform the SVD."""
        return self.__x

    @X.setter
    def X(self, matrix):
        self.__x = matrix

    @property
    def n_rows(self):
        """The number of rows in `X`."""
        return self.__n

    @property
    def n_columns(self):
        """The number of columns in `X`."""
        return self.__p

    @property
    def n_components(self):
        """The number of principal components that are calculated."""
        return self.__k

    @property
    def total_inertia(self):
        """The total inertia can be obtained differently for each kind of component analysis."""
        raise NotImplementedError

    @property
    def eigenvalues(self):
        """The eigenvalues associated to each principal component.

        The eigenvalues are obtained by squaring the singular values obtained from a SVD.

        Returns:
            List[float]: The eigenvalues ordered increasingly.
        """
        return np.square(self.svd.s).tolist()

    @property
    def explained_inertia(self):
        """The percentage of explained inertia per principal component.

        The explained inertia is obtained by dividing each eigenvalue by the total inertia.

        Returns:
            List[float]: The explained inertias ordered increasingly.
        """
        return [eig / self.total_inertia for eig in self.eigenvalues]

    @property
    def cumulative_explained_inertia(self):
        """The cumulative percentage of explained inertia per principal component.

        Returns:
            List[float]: The cumulative explained inertias ordered increasingly.
        """
        return np.cumsum(self.explained_inertia).tolist()

    def plot_inertia(self):
        """Plot a Scree diagram of the explained inertia per column."""
        return self.plotter.inertia(explained_inertia=self.explained_inertia)

    def plot_cumulative_inertia(self, threshold=0.8):
        """Plot a Scree diagram of the cumulative explained inertia per column.

        Args:
            threshold (float): The threshold at which a vertical line should be drawn. This is
                useful for finding the number of principal components required to reach a certain
                amount of cumulative inertia.
        """
        return self.plotter.cumulative_inertia(
            cumulative_explained_inertia=self.cumulative_explained_inertia,
            threshold=threshold
        )
