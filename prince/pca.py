"""Principal Component Analysis (PCA)"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import preprocessing
from sklearn import utils

from . import plot
from . import util
from . import svd


class PCA(base.BaseEstimator, base.TransformerMixin):

    """
    Args:
        rescale_with_mean (bool): Whether to substract each column's mean or not.
        rescale_with_std (bool): Whether to divide each column by it's standard deviation or not.
        n_components (int): The number of principal components to compute.
        n_iter (int): The number of iterations used for computing the SVD.
        copy (bool): Whether to perform the computations inplace or not.
    """

    def __init__(self, rescale_with_mean=True, rescale_with_std=True, n_components=2, n_iter=3,
                 copy=True, random_state=None, engine='auto'):

        self.n_components = n_components
        self.n_iter = n_iter
        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std = rescale_with_std
        self.copy = copy
        self.random_state = random_state
        self.engine = engine

    def fit(self, X, y=None):

        # Check input
        utils.check_array(X)

        # Convert pandas DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Copy data
        if self.copy:
            X = np.copy(X)

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
        self.total_inertia_ = np.sum(np.square(X))

        return self

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.
        """
        utils.validation.check_is_fitted(self, 's_')
        utils.check_array(X)
        return self.row_coordinates(X)

    def row_coordinates(self, X):
        """Returns the row principal coordinates.

        The row principal coordinates are obtained by projecting `X` on the right eigenvectors.
        """
        utils.validation.check_is_fitted(self, 's_')

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None

        # Copy data
        if self.copy:
            X = np.copy(X)

        # Scale data
        if hasattr(self, 'scaler_'):
            X = self.scaler_.transform(X)

        return pd.DataFrame(data=X.dot(self.V_.T), index=index)

    def row_standard_coordinates(self, X):
        """Returns the row standard coordinates.

        The row standard coordinates are obtained by dividing each row principal coordinate by it's
        associated eigenvalue.
        """
        utils.validation.check_is_fitted(self, 's_')
        return self.row_coordinates(X).div(self.eigenvalues_, axis='columns')

    def row_contributions(self, X):
        """Returns the row contributions towards each principal component.

        Each row contribution towards each principal component is equivalent to the amount of
        inertia it contributes. This is calculated by dividing the squared row coordinates by the
        eigenvalue associated to each principal component.
        """
        utils.validation.check_is_fitted(self, 's_')
        return np.square(self.row_coordinates(X)).div(self.eigenvalues_, axis='columns')

    def row_cosine_similarities(self, X):
        """Returns the cosine similarities between the rows and their principal components.

        The row cosine similarities are obtained by calculating the cosine of the angle shaped by
        the row principal coordinates and the row principal components. This is calculated by
        squaring each row projection coordinate and dividing each squared coordinate by the sum of
        the squared coordinates, which results in a ratio comprised between 0 and 1 representing the
        squared cosine.
        """
        utils.validation.check_is_fitted(self, 's_')
        squared_coordinates = np.square(self.row_coordinates(X))
        total_squares = squared_coordinates.sum(axis='columns')
        return squared_coordinates.div(total_squares, axis='rows')

    def column_correlations(self, X):
        """Returns the column correlations with each principal component."""
        utils.validation.check_is_fitted(self, 's_')

        # Convert pandas DataFrame to numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        row_pc = self.row_coordinates(X)

        return pd.DataFrame({
            component: {
                feature: row_pc[component].corr(X[feature])
                for feature in X.columns
            }
            for component in row_pc.columns
        })

    @property
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        utils.validation.check_is_fitted(self, 's_')
        return np.square(self.s_).tolist()

    @property
    def explained_inertia_(self):
        """Returns the percentage of explained inertia per principal component."""
        utils.validation.check_is_fitted(self, 's_')
        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    def plot_row_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                             labels=None, color_labels=None, ellipse_outline=False,
                             ellipse_fill=True, show_points=True, **kwargs):
        """Plot the row principal coordinates."""
        utils.validation.check_is_fitted(self, 's_')

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = plot.stylize_axis(ax)

        # Make sure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Retrieve principal coordinates
        coordinates = self.row_coordinates(X)
        x = coordinates[x_component]
        y = coordinates[y_component]

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
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]))

        # Legend
        ax.legend()

        # Text
        ax.set_title('Row principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax
