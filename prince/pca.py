"""Principal Component Analysis (PCA)"""
import functools

import numpy as np
import pandas as pd
import altair as alt
import sklearn.base
from sklearn import preprocessing
import sklearn.utils

from prince import plot
from prince import svd
from prince import utils


def select_active_variables(method):
    @functools.wraps(method)
    def _impl(self, X=None, *method_args, **method_kwargs):
        if hasattr(self, "feature_names_in_") and isinstance(X, pd.DataFrame):
            return method(
                self, X[self.feature_names_in_], *method_args, **method_kwargs
            )
        return method(self, X, *method_args, **method_kwargs)

    return _impl


class PCA(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, utils.EigenvaluesMixin
):
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
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std = rescale_with_std
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine

    def _check_input(self, X):
        if self.check_input:
            sklearn.utils.check_array(X)

    def fit(self, X, y=None, supplementary_columns=None):

        self._check_input(X)

        supplementary_columns = supplementary_columns or []
        active_variables = X.columns.difference(
            supplementary_columns, sort=False
        ).tolist()

        # https://scikit-learn.org/stable/developers/develop.html#universal-attributes
        self.feature_names_in_ = active_variables
        self.n_features_in_ = len(active_variables)

        X_active = X[active_variables].to_numpy(dtype=np.float64, copy=self.copy)
        if supplementary_columns:
            X_sup = X[supplementary_columns].to_numpy(dtype=np.float64, copy=self.copy)

        # Scale datarow_contributions
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=self.copy,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std,
            ).fit(X_active)
            X_active = self.scaler_.transform(
                X_active
            )  # TODO: maybe fit_transform is faster
            if supplementary_columns:
                X_sup = preprocessing.StandardScaler(
                    copy=self.copy,
                    with_mean=self.rescale_with_mean,
                    with_std=self.rescale_with_std,
                ).fit_transform(X_sup)

        self._column_dist = pd.Series(
            (X_active**2 / len(X_active)).sum(axis=0), index=active_variables
        )
        if supplementary_columns:
            self._column_dist = pd.concat(
                (
                    self._column_dist,
                    pd.Series(
                        (X_sup**2 / len(X_sup)).sum(axis=0),
                        index=supplementary_columns,
                    ),
                )
            )

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
        if supplementary_columns:
            self.column_coordinates_ = pd.concat(
                [
                    self.column_coordinates_,
                    pd.DataFrame(
                        data=X_sup.T @ (self.svd_.U / len(self.svd_.U) ** 0.5),
                        index=supplementary_columns,
                    ),
                ]
            )
        self.column_coordinates_.columns.name = "component"
        self.column_coordinates_.index.name = "variable"
        row_coords = pd.DataFrame(
            (self.svd_.U * len(self.svd_.U) ** 0.5) * self.eigenvalues_**0.5,
            # HACK: there's a circular dependency between row_contributions_
            # and active_row_coordinates in self.__init__
            index=self.row_contributions_.index
            if hasattr(self, "row_contributions_")
            else None,
        )
        row_coords.columns.name = "component"
        self.row_contributions_ = (row_coords**2 / len(X)).div(
            self.eigenvalues_, axis=1
        )
        self.row_contributions_.index = X.index

        return self

    @property
    @utils.check_is_fitted
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        return np.square(self.svd_.s) / len(self.svd_.U)

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

    @utils.check_is_fitted
    @select_active_variables
    def row_coordinates(self, X: pd.DataFrame):
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

    @utils.check_is_fitted
    def transform(self, X, as_array=False):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. This is just for compatibility with
        scikit-learn.

        """
        self._check_input(X)
        rc = self.row_coordinates(X)
        return rc.to_numpy() if as_array else rc

    def fit_transform(self, X, as_array=False):
        """A faster way to fit/transform.

        This methods produces exactly the same result as calling `fit(X)` followed
        by `transform(X)`. It is however much faster, as it avoids a matrix multiplication
        between the input data and the right eigenvectors. The row coordinates are instead obtained
        directly from the left eigenvectors.

        """
        self._check_input(X)
        self.fit(X)
        rc = self.row_coordinates(X)
        return rc.to_numpy() if as_array else rc

    @utils.check_is_fitted
    def inverse_transform(self, X, as_array=False):
        """Transforms row projections back to their original space.

        In other words, return a dataset whose transform would be X.

        """

        X_inv = np.dot(X, self.svd_.V)

        if hasattr(self, "scaler_"):
            X_inv = self.scaler_.inverse_transform(X_inv)

        if as_array:
            return X_inv

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(data=X_inv, index=index)

    @utils.check_is_fitted
    def row_standard_coordinates(self, X: pd.DataFrame = None):
        """Returns the row standard coordinates.

        The row standard coordinates are obtained by dividing each row principal coordinate by it's
        associated eigenvalue.

        """
        return self.row_coordinates(X).div(self.eigenvalues_, axis="columns")

    @utils.check_is_fitted
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

    @property
    @utils.check_is_fitted
    def column_correlations(self):
        """Calculate correlations between variables and components.

        The correlation between a variable and a component estimates the information they share. In
        the PCA framework, this correlation is called a loading.

        Note that the sum of the squared coefficients of correlation between a variable and all the
        components is equal to 1. As a consequence, the squared loadings are easier to interpret
        than the loadings (because the squared loadings give the proportion of the variance of the
        variables explained by the components).

        """
        return self.column_coordinates_.div(self._column_dist**0.5, axis=0)

    @property
    @utils.check_is_fitted
    def column_cosine_similarities_(self):
        return self.column_correlations**2

    @property
    @utils.check_is_fitted
    def column_contributions_(self):
        return (self.column_coordinates_.loc[self.feature_names_in_] ** 2).div(
            self.eigenvalues_, axis=1
        )

    @utils.check_is_fitted
    def plot(
        self,
        X,
        x_component=0,
        y_component=1,
        color_by=None,
        show_rows=True,
        show_columns=True,
        **params,
    ):

        if color_by is not None:
            params["color"] = color_by

        params["tooltip"] = (
            X.index.names
            if isinstance(X.index, pd.MultiIndex)
            else [X.index.name or "index"]  # index is the default name
        ) + [
            f"component {x_component}",
            f"component {y_component}",
        ]

        eig = self._eigenvalues_summary.to_dict(orient="index")
        chart = alt.LayerChart()

        if show_rows:
            row_coords = self.row_coordinates(X)
            row_coords.columns = [f"component {i}" for i in row_coords.columns]
            row_coords = row_coords.reset_index()
            row_chart = (
                alt.Chart(row_coords)
                .mark_circle(size=50)
                .encode(
                    alt.X(
                        f"component {x_component}",
                        scale=alt.Scale(zero=False),
                        axis=alt.Axis(
                            title=f"component {x_component} — {eig[x_component]['% of variance'] / 100:.2%}"
                        ),
                    ),
                    alt.Y(
                        f"component {y_component}",
                        scale=alt.Scale(zero=False),
                        axis=alt.Axis(
                            title=f"component {y_component} — {eig[y_component]['% of variance'] / 100:.2%}"
                        ),
                    ),
                    **params,
                )
            )
            chart += row_chart

        if show_columns:
            col_coords = self.column_coordinates_.copy()
            col_coords.columns = [f"component {i}" for i in col_coords.columns]
            col_coords = col_coords.reset_index()
            col_chart = (
                alt.Chart(col_coords)
                .mark_square(color="green", size=50)
                .encode(
                    alt.X(f"component {x_component}", scale=alt.Scale(zero=False)),
                    alt.Y(f"component {y_component}", scale=alt.Scale(zero=False)),
                    tooltip=["variable"],
                )
            )
            chart += col_chart

        return chart.interactive()
