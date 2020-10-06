"""Principal Component Analysis (PCA)"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import preprocessing
from sklearn import utils

from . import plot
from . import svd


class PCA(base.BaseEstimator, base.TransformerMixin):
    """Principal Component Analysis (PCA).

    Parameters:
        rescale_with_mean (bool): Whether to substract each column's mean or not.
        rescale_with_std (bool): Whether to divide each column by it's standard deviation or not.
        n_components (int): The number of principal components to compute.
        n_iter (int): The number of iterations used for computing the SVD.
        copy (bool): Whether to perform the computations inplace or not.
        check_input (bool): Whether to check the consistency of the inputs or not.
        as_array (bool): Whether to output an ``numpy.ndarray`` instead of a ``pandas.DataFrame``
            in ``tranform`` and ``inverse_transform``.

    """

    def __init__(self, rescale_with_mean=True, rescale_with_std=True, n_components=2, n_iter=3,
                 copy=True, check_input=True, random_state=None, engine='auto', as_array=False):
        self.n_components = n_components
        self.n_iter = n_iter
        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std = rescale_with_std
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine
        self.as_array = as_array

    def fit(self, X, y=None):

        # Check input
        if self.check_input:
            utils.check_array(X)

        # Convert pandas DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        # scikit-learn SLEP010
        self.n_features_in_ = X.shape[1]

        # Scale data
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=False,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std
            ).fit(X)
            X = self.scaler_.transform(X)

        # Compute SVD
        self.U_, self.s_, self.V_ = svd.compute_svd(
            X=X,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )

        # Compute total inertia
        self.total_inertia_ = np.sum(np.square(X)) / len(X)

        return self

    def _check_is_fitted(self):
        utils.validation.check_is_fitted(self, 'total_inertia_')

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.

        """
        self._check_is_fitted()
        if self.check_input:
            utils.check_array(X)
        rc = self.row_coordinates(X)
        if self.as_array:
            return rc.to_numpy()
        return rc

    def inverse_transform(self, X):
        """Transforms row projections back to their original space.

        In other words, return a dataset whose transform would be X.

        """
        self._check_is_fitted()
        X_inv = np.dot(X, self.V_)

        if hasattr(self, 'scaler_'):
            X_inv = self.scaler_.inverse_transform(X_inv)

        if self.as_array:
            return X_inv

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(data=X_inv, index=index)

    def row_coordinates(self, X):
        """Returns the row principal coordinates.

        The row principal coordinates are obtained by projecting `X` on the right eigenvectors.

        """
        self._check_is_fitted()

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        # Scale data
        if hasattr(self, 'scaler_'):
            X = self.scaler_.transform(X)

        return pd.DataFrame(data=X.dot(self.V_.T), index=index, dtype=np.float64)

    def row_standard_coordinates(self, X):
        """Returns the row standard coordinates.

        The row standard coordinates are obtained by dividing each row principal coordinate by it's
        associated eigenvalue.

        """
        self._check_is_fitted()
        return self.row_coordinates(X).div(self.eigenvalues_, axis='columns')

    def row_contributions(self, X):
        """Returns the row contributions towards each principal component.

        Each row contribution towards each principal component is equivalent to the amount of
        inertia it contributes. This is calculated by dividing the squared row coordinates by the
        eigenvalue associated to each principal component.

        """
        self._check_is_fitted()
        return np.square(self.row_coordinates(X)).div(self.eigenvalues_, axis='columns')

    def row_cosine_similarities(self, X):
        """Returns the cosine similarities between the rows and their principal components.

        The row cosine similarities are obtained by calculating the cosine of the angle shaped by
        the row principal coordinates and the row principal components. This is calculated by
        squaring each row projection coordinate and dividing each squared coordinate by the sum of
        the squared coordinates, which results in a ratio comprised between 0 and 1 representing the
        squared cosine.

        """
        self._check_is_fitted()
        squared_coordinates = np.square(self.row_coordinates(X))
        total_squares = squared_coordinates.sum(axis='columns')
        return squared_coordinates.div(total_squares, axis='rows')

    def column_correlations(self, X):
        """Returns the column correlations with each principal component."""
        self._check_is_fitted()

        # Convert numpy array to pandas DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        row_pc = self.row_coordinates(X)

        return pd.DataFrame({
            component: {
                feature: row_pc[component].corr(X[feature])
                for feature in X.columns
            }
            for component in row_pc.columns
        }).sort_index()

    @property
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        self._check_is_fitted()
        return np.square(self.s_) / len(self.U_)

    @property
    def explained_inertia_(self):
        """Returns the percentage of explained inertia per principal component."""
        return self.eigenvalues_ / self.total_inertia_

    def plot_row_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                             labels=None, color_labels=None, ellipse_outline=False,
                             ellipse_fill=True, show_points=True, **kwargs):
        """Plot the row principal coordinates."""
        self._check_is_fitted()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = plot.stylize_axis(ax)

        # Make sure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Retrieve principal coordinates
        coordinates = self.row_coordinates(X)
        x = coordinates[x_component].astype(np.float)
        y = coordinates[y_component].astype(np.float)

        # Plot
        if color_labels is None:
            ax.scatter(x, y, **kwargs)
        else:
            for color_label in sorted(list(set(color_labels))):
                mask = np.array(color_labels) == color_label
                color = ax._get_lines.get_next_color()
                # Plot points
                if show_points:
                    ax.scatter(x[mask], y[mask], color=color, **kwargs, label=color_label)
                # Plot ellipse
                if (ellipse_outline or ellipse_fill):
                    x_mean, y_mean, width, height, angle = plot.build_ellipse(x[mask], y[mask])
                    ax.add_patch(mpl.patches.Ellipse(
                        (x_mean, y_mean),
                        width,
                        height,
                        angle=angle,
                        linewidth=2 if ellipse_outline else 0,
                        color=color,
                        fill=ellipse_fill,
                        alpha=0.2 + (0.3 if not show_points else 0) if ellipse_fill else 1
                    ))

        # Add labels
        if labels is not None:
            for xi, yi, label in zip(x, y, labels):
                ax.annotate(label, (xi, yi))

        # Legend
        ax.legend()

        # Text
        ax.set_title('Row principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax
