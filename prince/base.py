import numpy as np
import pandas as pd


class Base():

    """Base contains the common operations performed during any factor analysis.

    Attributes:
        plotter (prince.Plotter): A plotter instance used for displaying charts. The plotter is
            defined by the type of component analysis and the plotting backend. The plotter is
            pre-configuredto display the charts usually associated with each kind of factorial
            analysis.
        svd (prince.SVD): The object containing the results from the Singular Value Decomposition.
        X (pandas.DataFrame): The dataframe on which was applied the SVD. For various (good)
            reasons, this dataframe can differ from the one provided by the user.
        n (int): The number of rows in `X`.
        p (int): The number of columns in `X`.
        k (int): The number of computed principal components, if inferior or equal to `p`.
        eigenvalues (list(float)): The list of eigenvalues resulting from the SVD operation.
        total_inertia (float): The total inertia, which is equivalent to the sum of the
            eigenvalues if they have all been calculated.
        explained_inertia (list(float)): The percentage of inertia associated to each principal
            component calculated by dividing the eigenvalues by the total inertia.
        cumulative_explained_inertia (list(float)): The cumsum of the explained inertia.
    """

    def __init__(self, dataframe, k, plotter):

        if plotter not in 'mpl':
            raise ValueError('Unrecognized plotting backend; choose from: mpl')

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError('dataframe muse be a pandas.DataFrame')


        self.plotter = None
        self.svd = None

        self.X = dataframe
        self.n, self.p = self.X.shape
        # Determine the number of components computed during SVD
        self.k = self.p if k == -1 else min(k, self.p)

    def _compute_svd(self):
        raise NotImplementedError

    def _set_plotter(self, plotter_name):
        raise NotImplementedError

    @property
    def total_inertia(self):
        """The total inertia can be obtained differently for each kind of component analysis."""
        raise NotImplementedError

    @property
    def eigenvalues(self):
        """The eigenvalues obtained by squaring the singular values obtained from an SVD."""
        return np.square(self.svd.s).tolist()

    @property
    def explained_inertia(self):
        """The percentage of explained inertia per principal component."""
        return [eig / self.total_inertia for eig in self.eigenvalues]

    @property
    def cumulative_explained_inertia(self):
        """The cumulative percentage of explained inertia per principal component."""
        return np.cumsum(self.explained_inertia).tolist()

    def plot_inertia(self):
        """Plot a Scree diagram of the explained inertia per column."""
        return self.plotter.inertia(explained_inertia=self.explained_inertia)

    def plot_cumulative_inertia(self, threshold=0.8):
        """Plot a Scree diagram of the cumulative explained inertia per column."""
        return self.plotter.cumulative_inertia(
            cumulative_explained_inertia=self.cumulative_explained_inertia,
            threshold=threshold
        )
