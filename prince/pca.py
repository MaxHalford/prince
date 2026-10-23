"""Principal Component Analysis (PCA)"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import altair as alt
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.utils
from sklearn import preprocessing
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from typing_extensions import override

from prince import svd, utils

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # scikit-learn 1.5

    def validate_data(estimator, X, *, reset, **check_params):
        return estimator._validate_data(X, reset=reset, **check_params)


if TYPE_CHECKING:
    import numpy.typing as npt


P = ParamSpec("P")
R = TypeVar("R")


def select_active_variables(method: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(method)
    def _impl(self: Any, X: Any = None, *method_args: Any, **method_kwargs: Any) -> R:
        if hasattr(self, "feature_names_in_"):
            X = self._select_columns(X, self.feature_names_in_)
        return method(self, X, *method_args, **method_kwargs)

    return cast(Callable[P, R], _impl)


class PCA(
    ClassNamePrefixFeaturesOutMixin,
    sklearn.base.TransformerMixin,
    sklearn.base.BaseEstimator,
    utils.EigenvaluesMixin,
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

    @override
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.transformer_tags is not None:
            tags.transformer_tags.preserves_dtype = []
        return tags

    @staticmethod
    def _as_narwhals_dataframe(X):
        """Return an eager Narwhals dataframe, or ``None`` for non-dataframe input."""
        if isinstance(X, pd.DataFrame):
            return None
        return utils.as_narwhals_dataframe(X)

    @classmethod
    def _column_names(cls, X):
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        if (frame := cls._as_narwhals_dataframe(X)) is not None:
            return frame.columns
        return None

    @classmethod
    def _select_columns(cls, X, columns):
        if isinstance(X, pd.DataFrame):
            return X[columns]
        if (frame := cls._as_narwhals_dataframe(X)) is not None:
            return frame.select(list(columns)).to_native()
        return X

    def _to_numpy(self, X, columns=None):
        if columns is not None:
            X = self._select_columns(X, columns)
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64, copy=self.copy)
        if (frame := self._as_narwhals_dataframe(X)) is not None:
            array = np.asarray(frame.to_numpy(), dtype=np.float64)
        else:
            array = np.asarray(X, dtype=np.float64)
        return array.copy() if self.copy else array

    @classmethod
    def _row_index(cls, X, n_rows=None):
        if isinstance(X, pd.DataFrame):
            return X.index
        if (frame := cls._as_narwhals_dataframe(X)) is not None:
            return pd.RangeIndex(frame.shape[0])
        if n_rows is None:
            n_rows = np.asarray(X).shape[0]
        return pd.RangeIndex(n_rows)

    def _validate_array(self, X, *, reset):
        if self.check_input:
            return validate_data(
                self,
                X,
                reset=reset,
                dtype=np.float64,
                copy=self.copy,
            )
        X = self._to_numpy(X)
        if reset:
            self.n_features_in_ = X.shape[1]
        return X

    def _validate_transform_input(self, X):
        if not self.check_input:
            return self._to_numpy(X)
        if not isinstance(X, pd.DataFrame) and self._as_narwhals_dataframe(X) is not None:
            X_array = sklearn.utils.check_array(self._to_numpy(X))
            if X_array.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X_array.shape[1]} features, but PCA is expecting "
                    f"{self.n_features_in_} features as input"
                )
            return X_array
        return validate_data(
            self,
            X,
            reset=False,
            dtype=np.float64,
            copy=self.copy,
        )

    def fit(
        self,
        X,
        y=None,
        sample_weight=None,
        column_weight=None,
        supplementary_columns=None,
    ):
        # Massage input
        supplementary_columns = supplementary_columns or []
        column_names = self._column_names(X)
        active_variables: list[Any] = []
        if column_names is None:
            if supplementary_columns:
                raise ValueError("supplementary_columns requires dataframe input")
            X_active = X
        else:
            supplementary_set = set(supplementary_columns)
            active_variables = [name for name in column_names if name not in supplementary_set]
            X_active = self._select_columns(X, active_variables)

        X_active = self._validate_array(X_active, reset=True)
        if column_names is None:
            active_variables = list(range(X_active.shape[1]))
        sample_weight = (
            np.ones(len(X_active), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        if sample_weight.shape != (len(X_active),):
            raise ValueError("sample_weight must be one-dimensional with one value per row")
        if not np.isfinite(sample_weight).all() or (sample_weight < 0).any():
            raise ValueError("sample_weight must contain only finite, non-negative values")
        if sample_weight.sum() == 0:
            raise ValueError("sample_weight must sum to a positive value")
        sample_weight = sample_weight / sample_weight.sum()
        column_weight = (
            np.ones(len(active_variables), dtype=np.float64)
            if column_weight is None
            else np.asarray(column_weight, dtype=np.float64)
        )
        if column_weight.shape != (len(active_variables),):
            raise ValueError("column_weight must have one value per active column")
        if not np.isfinite(column_weight).all() or (column_weight < 0).any():
            raise ValueError("column_weight must contain only finite, non-negative values")
        self.column_weight_ = column_weight

        # https://scikit-learn.org/stable/developers/develop.html#universal-attributes
        self._active_variables_ = active_variables
        if column_names is not None:
            self.feature_names_in_ = np.empty(len(active_variables), dtype=object)
            self.feature_names_in_[:] = active_variables
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_features_in_ = len(active_variables)

        X_sup: npt.NDArray[np.float64] = np.empty((len(X_active), 0), dtype=np.float64)
        if supplementary_columns:
            X_sup = self._to_numpy(X, supplementary_columns)

        # Scale datarow_contributions
        if hasattr(self, "scaler_"):
            del self.scaler_
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=self.copy,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std,
            ).fit(X_active, sample_weight=sample_weight)
            X_active = self.scaler_.transform(X_active)  # TODO: maybe fit_transform is faster
            if supplementary_columns:
                X_sup = preprocessing.StandardScaler(
                    copy=self.copy,
                    with_mean=self.rescale_with_mean,
                    with_std=self.rescale_with_std,
                ).fit_transform(X_sup, sample_weight=sample_weight)

        self._column_dist = pd.Series(
            sample_weight @ np.square(X_active),
            index=active_variables,
        )
        if supplementary_columns:
            self._column_dist = pd.concat(
                (
                    self._column_dist,
                    pd.Series(
                        sample_weight @ np.square(X_sup),
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
            row_weights=sample_weight,
            column_weights=column_weight,
        )
        self._n_features_out = len(self.svd_.s)

        self.total_inertia_ = sample_weight @ np.square(X_active) @ np.asarray(column_weight)

        self.column_coordinates_ = pd.DataFrame(
            data=self.svd_.V.T * self.eigenvalues_**0.5,
            index=active_variables,
        )
        if supplementary_columns:
            self.column_coordinates_ = pd.concat(
                [
                    self.column_coordinates_,
                    pd.DataFrame(
                        data=X_sup.T @ (sample_weight[:, np.newaxis] * self.svd_.U),
                        index=supplementary_columns,
                    ),
                ]
            )
        self.column_coordinates_.columns.name = "component"
        self.column_coordinates_.index.name = "variable"
        row_coords = pd.DataFrame(
            self.svd_.U * self.eigenvalues_**0.5,
            index=self._row_index(X, len(X_active)),
        )
        row_coords.columns.name = "component"
        self.row_contributions_ = (row_coords**2 * sample_weight[:, np.newaxis]).div(
            self.eigenvalues_, axis=1
        )
        return self

    @property
    @utils.check_is_fitted
    @override
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        return np.square(self.svd_.s)

    def _scale(self, X):
        X_array = self._validate_transform_input(X)
        return self.scaler_.transform(X_array) if hasattr(self, "scaler_") else X_array

    @utils.check_is_fitted
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

        index = self._row_index(X)
        X_arr = np.array(self._scale(X), copy=self.copy)
        X_arr *= self.column_weight_

        coord = pd.DataFrame(data=X_arr.dot(self.svd_.V.T), index=index)
        coord.columns.name = "component"
        return coord

    @utils.check_is_fitted
    def transform(self, X, as_array=False):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. This is just for compatibility with
        scikit-learn.

        """
        rc = self.row_coordinates(X)
        return rc.to_numpy() if as_array else rc

    @override
    def fit_transform(self, X, y=None, as_array=False, **fit_params):
        """A faster way to fit/transform.

        This methods produces exactly the same result as calling `fit(X)` followed
        by `transform(X)`. It is however much faster, as it avoids a matrix multiplication
        between the input data and the right eigenvectors. The row coordinates are instead obtained
        directly from the left eigenvectors.

        """
        self.fit(X, y, **fit_params)
        rc = self.row_coordinates(X)
        return rc.to_numpy() if as_array else rc

    @utils.check_is_fitted
    def inverse_transform(self, X, as_array=False):
        """Transforms row projections back to their original space.

        In other words, return a dataset whose transform would be X.

        """

        index = X.index if isinstance(X, pd.DataFrame) else None
        X_inv = np.dot(self._to_numpy(X), self.svd_.V)

        if hasattr(self, "scaler_"):
            X_inv = self.scaler_.inverse_transform(X_inv)

        if as_array:
            return X_inv

        columns = self._active_variables_ if hasattr(self, "feature_names_in_") else None
        return pd.DataFrame(data=X_inv, index=index, columns=columns)

    @utils.check_is_fitted
    def row_standard_coordinates(self, X):
        """Returns the row standard coordinates.

        The row standard coordinates are obtained by dividing each row principal coordinate
        by the square root of the associated eigenvalue.

        """
        return self.row_coordinates(X).div(self.eigenvalues_**0.5, axis="columns")

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
        scaled = np.asarray(self._scale(X), dtype=np.float64)
        squared_coordinates = (scaled * scaled) @ np.asarray(self.column_weight_)
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
        return (
            ((self.column_coordinates_.loc[self._active_variables_]) ** 2)
            * self.column_weight_[:, np.newaxis]
        ).div(self.eigenvalues_, axis=1)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def plot(
        self,
        X,
        x_component=0,
        y_component=1,
        color_rows_by=None,
        show_row_markers=True,
        show_column_markers=True,
        show_row_labels=False,
        show_column_labels=False,
        row_labels_column=None,
    ):
        row_params: dict[str, Any] = {
            "tooltip": (
                X.index.names
                if isinstance(X.index, pd.MultiIndex)
                else [X.index.name or "index"]  # index is the default name
            )
            + [
                f"component {x_component}",
                f"component {y_component}",
            ]
        }
        if color_rows_by:
            row_params["color"] = color_rows_by

        eig = self._eigenvalues_summary.to_dict(orient="index")

        row_chart_markers = None
        row_chart_labels = None
        column_chart_markers = None
        column_chart_labels = None

        # Row coordinates are needed both to draw the row chart and to scale the column
        # coordinates into the same space for the biplot, so they are computed once
        # whenever either chart is drawn.
        if show_row_markers or show_row_labels or show_column_markers or show_column_labels:
            row_coords = self.row_coordinates(X)
            row_coords.columns = [f"component {i}" for i in row_coords.columns]

            if show_row_markers or show_row_labels:
                row_labels = (
                    pd.Series(
                        row_coords.index.get_level_values(
                            row_labels_column or row_coords.index.names[0]
                        ),
                        index=row_coords.index,
                    )
                    if isinstance(row_coords.index, pd.MultiIndex)
                    else pd.Series(row_coords.index, index=row_coords.index)
                )

                row_chart = alt.Chart(row_coords.assign(label=row_labels).reset_index()).encode(
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
                    **row_params,
                )
                row_chart_markers = row_chart.mark_circle(size=50 if show_row_markers else 0)
                if show_row_labels:
                    row_chart_labels = row_chart.mark_text().encode(text="label:N")

            if show_column_markers or show_column_labels:
                column_coords = self.column_coordinates_.copy()
                column_coords.columns = [f"component {i}" for i in column_coords.columns]
                # Scale the column coordinates into the row coordinate space for the biplot.
                column_coords = column_coords * row_coords.abs().max()
                column_labels = pd.Series(column_coords.index, index=column_coords.index)

                column_chart = alt.Chart(
                    column_coords.assign(label=column_labels).reset_index()
                ).encode(
                    alt.X(f"component {x_component}", scale=alt.Scale(zero=False)),
                    alt.Y(f"component {y_component}", scale=alt.Scale(zero=False)),
                    tooltip=["variable"],
                )
                column_chart_markers = column_chart.mark_square(
                    color="green", size=50 if show_column_markers else 0
                )
                if show_column_labels:
                    column_chart_labels = column_chart.mark_text().encode(text="label:N")

        charts = filter(
            None,
            (
                row_chart_markers,
                row_chart_labels,
                column_chart_markers,
                column_chart_labels,
            ),
        )

        return alt.layer(*charts).interactive()
