"""Principal Component Analysis (PCA)"""
import functools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import preprocessing
from sklearn import utils

from . import plot
from . import svd


def check_is_fitted(method):
    @functools.wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        self._check_is_fitted()
        return method(self, *method_args, **method_kwargs)

    return _impl


def select_active_variables(method):
    @functools.wraps(method)
    def _impl(self, X, *method_args, **method_kwargs):
        if hasattr(self, "feature_names_in_") and isinstance(X, pd.DataFrame):
            return method(
                self, X[self.feature_names_in_], *method_args, **method_kwargs
            )
        return method(self, X, *method_args, **method_kwargs)

    return _impl


class PCA(base.BaseEstimator, base.TransformerMixin):
    """Principal Component Analysis (PCA).

    Parameters
    ----------
    rescale_with_mean
        Whether or not to subtract each column's mean before performing SVD.
    rescale_with_std
        Whether or not to standardize each column before performing SVD.
    n_components
        The number of principal components to compute.
    n_iter
        The number of iterations used for computing the SVD.
    copy
        Whether nor to perform the computations inplace.
    check_input
        Whether to check the coherence of the inputs or not.

    """

    def __init__(
        self,
        rescale_with_mean=True,
        rescale_with_std=True,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        as_array=False,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std = rescale_with_std
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine
        self.as_array = as_array

    def fit(self, X, y=None, supplementary_variables=None):

        supplementary_variables = supplementary_variables or []
        active_variables = X.columns.difference(
            supplementary_variables, sort=False
        ).tolist()

        # https://scikit-learn.org/stable/developers/develop.html#universal-attributes
        self.feature_names_in_ = active_variables
        self.n_features_in_ = len(active_variables)

        X_active = X[active_variables].to_numpy(dtype=np.float64, copy=self.copy)
        if supplementary_variables:
            X_sup = X[supplementary_variables].to_numpy(
                dtype=np.float64, copy=self.copy
            )
        # X = X.to_numpy(dtype=np.float64, copy=self.copy)
        # self._check_input(X)

        # Scale data
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=self.copy,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std,
            ).fit(X_active)
            X_active = self.scaler_.transform(
                X_active
            )  # TODO: maybe fit_transform is faster
            if supplementary_variables:
                X_sup = preprocessing.StandardScaler(
                    copy=self.copy,
                    with_mean=self.rescale_with_mean,
                    with_std=self.rescale_with_std,
                ).fit_transform(X_sup)

        self.svd_ = svd.compute_svd(
            X=X_active,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine,
        )

        self.total_inertia_ = np.sum(np.square(X_active)) / len(X_active)

        self.column_coordinates_ = pd.DataFrame(
            data=self.svd_.V.T * self.eigenvalues_**0.5,
            index=active_variables,
        )
        if supplementary_variables:
            self.column_coordinates_ = pd.concat(
                [
                    self.column_coordinates_,
                    pd.DataFrame(
                        data=X_sup.T @ (self.svd_.U / len(self.svd_.U) ** 0.5),
                        index=supplementary_variables,
                    ),
                ]
            )
        self.column_coordinates_.columns.name = "component"
        self.row_contributions_ = (self.active_row_coordinates**2 / len(X)).div(
            self.eigenvalues_, axis=1
        )
        self.row_contributions_.index = X.index

        return self

    def _check_is_fitted(self):
        utils.validation.check_is_fitted(self, "total_inertia_")

    def _check_input(self, X):
        if self.check_input:
            utils.check_array(X)

    def _scale(self, X):

        if not hasattr(self, "scaler_"):
            return X

        if sup_variables := X.columns.difference(
            self.feature_names_in_, sort=False
        ).tolist():
            X = np.concatenate(
                (
                    self.scaler_.transform(X[self.feature_names_in_].to_numpy()),
                    preprocessing.StandardScaler(
                        copy=self.copy,
                        with_mean=self.rescale_with_mean,
                        with_std=self.rescale_with_std,
                    ).fit_transform(X[sup_variables]),
                ),
                axis=1,
            )
        else:
            X = self.scaler_.transform(X.to_numpy())

        return X

    @property
    @check_is_fitted
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        return np.square(self.svd_.s) / len(self.svd_.U)

    @property
    @check_is_fitted
    def percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        return self.eigenvalues_ / self.total_inertia_

    @property
    @check_is_fitted
    def cumulative_percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        return np.cumsum(self.percentage_of_variance_)

    @property
    @check_is_fitted
    def eigenvalues_summary(self):
        """Returns the eigenvalues associated with each principal component."""
        summary = pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues_,
                r"% of variance": self.percentage_of_variance_,
                r"% of variance (cumulative)": self.cumulative_percentage_of_variance_,
            }
        ).style.format(
            {
                "eigenvalue": "{:,.3f}".format,
                "% of variance": "{:,.2%}".format,
                "% of variance (cumulative)": "{:,.2%}".format,
            }
        )
        summary.index.name = "component"
        return summary

    @check_is_fitted
    @select_active_variables
    def row_coordinates(self, X):
        """Returns the row principal coordinates.

        The row principal coordinates are obtained by projecting `X` on the right eigenvectors.

        Synonyms
        --------
        Row projections
        Factor scores
        Loadings

        """
        index = X.index if isinstance(X, pd.DataFrame) else None
        X = self._scale(X)
        X = np.array(X, copy=self.copy)

        coord = pd.DataFrame(data=X.dot(self.svd_.V.T), index=index)
        coord.columns.name = "component"
        return coord

    @check_is_fitted
    def transform(self, X, as_array=False):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. This is just for compatibility with
        scikit-learn.

        """
        self._check_input(X)
        rc = self.row_coordinates(X)
        if as_array:
            return rc.to_numpy()
        return rc

    @property
    @check_is_fitted
    def active_row_coordinates(self):
        coords = pd.DataFrame(
            (self.svd_.U * len(self.svd_.U) ** 0.5) * self.eigenvalues_**0.5,
            index=self.row_contributions.index
            if hasattr(self, "row_contributions")
            else None,
        )
        coords.columns.name = "component"
        return coords

    @check_is_fitted
    def fit_transform(self, X, as_array=False):
        """A faster way to fit/transform.

        This methods produces exactly the same result as calling `fit(X)` followed
        by `transform(X)`. It is however much faster, as it avoids a matrix multiplication
        between the input data and the right eigenvectors. The row coordinates are instead obtained
        directly from the left eigenvectors.

        """
        self.fit(X)
        return self.active_row_coordinates

    @check_is_fitted
    def inverse_transform(self, X):
        """Transforms row projections back to their original space.

        In other words, return a dataset whose transform would be X.

        """

        X_inv = np.dot(X, self.V_)

        if hasattr(self, "scaler_"):
            X_inv = self.scaler_.inverse_transform(X_inv)

        if self.as_array:
            return X_inv

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(data=X_inv, index=index)

    @check_is_fitted
    def row_standard_coordinates(self, X):
        """Returns the row standard coordinates.

        The row standard coordinates are obtained by dividing each row principal coordinate by it's
        associated eigenvalue.

        """
        return self.row_coordinates(X).div(self.eigenvalues_, axis="columns")

    @check_is_fitted
    @select_active_variables
    def row_cosine_similarities(self, X):
        """Returns the cosine similarities between the rows and their principal components.

        The row cosine similarities are obtained by calculating the cosine of the angle shaped by
        the row principal coordinates and the row principal components. This is calculated by
        squaring each row projection coordinate and dividing each squared coordinate by the sum of
        the squared coordinates, which results in a ratio comprised between 0 and 1 representing
        the squared cosine.

        """
        squared_coordinates = np.square(self._scale(X)).sum(axis=1)
        return (self.row_coordinates(X) ** 2).div(squared_coordinates, axis=0)

    # @check_is_fitted
    # def row_contributions(self, X):
    #     """Returns the row contributions towards each principal component.

    #     The eigenvalue associated to a component is equal to the sum of the squared factor scores
    #     for this component. Therefore, the importance of an observation for a component can be
    #     obtained by the ratio of the squared factor score of this observation by the eigenvalue
    #     associated with that component. This ratio is called the contribution of the observation to
    #     the component.

    #     The value of a contribution is between 0 and 1 and, for a given component, the sum of the
    #     contributions of all observations is equal to 1. The larger the value of the contribution,
    #     the more the observation contributes to the component. A useful heuristic is to base the
    #     interpretation of a component on the observations whose contribution is larger than the
    #     average contribution (i.e., observations whose contribution is larger than `1 / n`). The
    #     observations with high contributions and different signs can then be opposed to help
    #     interpret the component because these observations represent the two endpoints of this
    #     component.

    #     """

    @property
    @check_is_fitted
    def column_correlations(self):
        """Calculate correlations between variables and components.

        The correlation between a variable and a component estimates the information they share. In
        the PCA framework, this correlation is called a loading.

        Note that the sum of the squared coefficients of correlation between a variable and all the
        components is equal to 1. As a consequence, the squared loadings are easier to interpret
        than the loadings (because the squared loadings give the proportion of the variance of the
        variables explained by the components).

        """
        return self.column_coordinates_

    @property
    @check_is_fitted
    def column_cosine_similarities(self):
        return self.column_correlations**2

    @property
    @check_is_fitted
    def column_contributions(self):
        return (self.column_coordinates_.loc[self.feature_names_in_] ** 2).div(
            self.eigenvalues_, axis=1
        )

    @check_is_fitted
    def plot_row_coordinates(
        self,
        X,
        ax=None,
        figsize=(6, 6),
        x_component=0,
        y_component=1,
        label_by=None,
        color_by=None,
        ellipse_outline=False,
        ellipse_fill=True,
        show_points=True,
        **kwargs,
    ):
        """Plot the row principal coordinates."""

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
        color_labels = None
        if color_by:
            try:
                color_labels = X.index.get_level_values(color_by)
            except ValueError:
                color_labels = X[color_by]

        if color_labels is None:
            ax.scatter(x, y, **kwargs)
        else:
            for color_label in sorted(list(set(color_labels))):
                mask = np.array(color_labels) == color_label
                color = ax._get_lines.get_next_color()
                # Plot points
                if show_points:
                    ax.scatter(
                        x[mask], y[mask], color=color, **kwargs, label=color_label
                    )
                # Plot ellipse
                if ellipse_outline or ellipse_fill:
                    x_mean, y_mean, width, height, angle = plot.build_ellipse(
                        x[mask], y[mask]
                    )
                    ax.add_patch(
                        mpl.patches.Ellipse(
                            (x_mean, y_mean),
                            width,
                            height,
                            angle=angle,
                            linewidth=2 if ellipse_outline else 0,
                            color=color,
                            fill=ellipse_fill,
                            alpha=0.2 + (0.3 if not show_points else 0)
                            if ellipse_fill
                            else 1,
                        )
                    )

        # Add labels
        labels = None
        if label_by:
            try:
                labels = X.index.get_level_values(label_by)
            except ValueError:
                labels = X[label_by]

        if labels is not None:
            for xi, yi, label in zip(x, y, labels):
                ax.annotate(label, (xi, yi))

        # Legend
        ax.legend()

        # Text
        ax.set_title("Row principal coordinates")
        ei = self.percentage_of_variance_
        ax.set_xlabel(
            "Component {} ({:.2f}% inertia)".format(x_component, 100 * ei[x_component])
        )
        ax.set_ylabel(
            "Component {} ({:.2f}% inertia)".format(y_component, 100 * ei[y_component])
        )

        return ax
