"""Correspondence Analysis (CA)"""

from __future__ import annotations

import functools

import altair as alt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils import check_array

from prince import svd, utils


def select_active_columns(method):
    @functools.wraps(method)
    def _impl(self, X=None, *method_args, **method_kwargs):
        if hasattr(self, "active_cols_") and isinstance(X, pd.DataFrame):
            return method(self, X[self.active_cols_], *method_args, **method_kwargs)
        return method(self, X, *method_args, **method_kwargs)

    return _impl


def select_active_rows(method):
    @functools.wraps(method)
    def _impl(self, X=None, *method_args, **method_kwargs):
        if hasattr(self, "active_rows_") and isinstance(X, pd.DataFrame):
            return method(self, X.loc[self.active_rows_], *method_args, **method_kwargs)
        return method(self, X, *method_args, **method_kwargs)

    return _impl


class CA(utils.EigenvaluesMixin):
    def __init__(
        self,
        n_components=2,
        n_iter=10,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine

    @utils.check_is_dataframe_input
    def fit(self, X, y=None):
        # Check input
        if self.check_input:
            check_array(X)

        # Check all values are positive
        if (X < 0).any().any():
            raise ValueError("All values in X should be positive")

        _, row_names, _, col_names = utils.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.copy:
            X = np.copy(X)

        # Compute the correspondence matrix which contains the relative frequencies
        X = X.astype(float) / np.sum(X)

        # Compute row and column masses
        self.row_masses_ = pd.Series(X.sum(axis=1), index=row_names)
        self.col_masses_ = pd.Series(X.sum(axis=0), index=col_names)

        self.active_rows_ = self.row_masses_.index.unique()
        self.active_cols_ = self.col_masses_.index.unique()

        # Compute standardised residuals
        r = self.row_masses_.to_numpy()
        c = self.col_masses_.to_numpy()
        S = sparse.diags(r**-0.5) @ (X - np.outer(r, c)) @ sparse.diags(c**-0.5)

        # Compute SVD on the standardised residuals
        self.svd_ = svd.compute_svd(
            X=S,
            n_components=min(self.n_components, min(X.shape) - 1),
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine,
        )

        # Compute total inertia
        self.total_inertia_ = np.einsum("ij,ji->", S, S.T)

        self.row_contributions_ = pd.DataFrame(
            sparse.diags(self.row_masses_.values)
            @ np.divide(
                # Same as row_coordinates(X)
                (
                    sparse.diags(self.row_masses_.values**-0.5)
                    @ self.svd_.U
                    @ sparse.diags(self.svd_.s)
                )
                ** 2,
                self.eigenvalues_,
                out=np.zeros((len(self.row_masses_), len(self.eigenvalues_))),
                where=self.eigenvalues_ > 0,
            ),
            index=self.row_masses_.index,
        )

        self.column_contributions_ = pd.DataFrame(
            sparse.diags(self.col_masses_.values)
            @ np.divide(
                # Same as col_coordinates(X)
                (
                    sparse.diags(self.col_masses_.values**-0.5)
                    @ self.svd_.V.T
                    @ sparse.diags(self.svd_.s)
                )
                ** 2,
                self.eigenvalues_,
                out=np.zeros((len(self.col_masses_), len(self.eigenvalues_))),
                where=self.eigenvalues_ > 0,
            ),
            index=self.col_masses_.index,
        )

        return self

    @property
    @utils.check_is_fitted
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        return np.square(self.svd_.s)

    @utils.check_is_dataframe_input
    @select_active_columns
    def row_coordinates(self, X):
        """The row principal coordinates."""

        _, row_names, _, _ = utils.make_labels_and_names(X)
        index_name = X.index.name

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Normalise the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.col_masses_.to_numpy() ** -0.5) @ self.svd_.V.T,
            index=pd.Index(row_names, name=index_name),
        )

    @utils.check_is_dataframe_input
    @select_active_columns
    def row_cosine_similarities(self, X):
        """Return the cos2 for each row against the dimensions.

        The cos2 value gives an indicator of the accuracy of the row projection on the dimension.

        Values above 0.5 usually means that the row is relatively accurately well projected onto that dimension. Its often
        used to identify which factor/dimension is important for a given element as the cos2 can be interpreted as the proportion
        of the variance of the element attributed to a particular factor.

        """
        F = self.row_coordinates(X)
        return self._row_cosine_similarities(X, F)

    @select_active_columns
    def _row_cosine_similarities(self, X, F):
        # Active
        X_act = X.loc[self.active_rows_]
        X_act = X_act / X_act.sum().sum()
        marge_col = X_act.sum(axis=0)
        Tc = X_act.div(X_act.sum(axis=1), axis=0).div(marge_col, axis=1) - 1
        dist2_row = (Tc**2).mul(marge_col, axis=1).sum(axis=1)

        # Supplementary
        X_sup = X.loc[X.index.difference(self.active_rows_, sort=False)]
        X_sup = X_sup.div(X_sup.sum(axis=1), axis=0)
        dist2_row_sup = ((X_sup - marge_col) ** 2).div(marge_col, axis=1).sum(axis=1)

        dist2_row = pd.concat((dist2_row, dist2_row_sup))

        # Can't use pandas.div method because it doesn't support duplicate indices
        return F**2 / dist2_row.to_numpy()[:, None]

    @utils.check_is_dataframe_input
    @select_active_rows
    def column_coordinates(self, X):
        """The column principal coordinates."""

        _, _, _, col_names = utils.make_labels_and_names(X)
        index_name = X.columns.name

        if isinstance(X, pd.DataFrame):
            is_sparse = X.dtypes.apply(lambda dtype: isinstance(dtype, pd.SparseDtype)).all()
            if is_sparse:
                X = X.sparse.to_coo()
            else:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Transpose and make sure the rows sum up to 1
        if isinstance(X, np.ndarray):
            X = X.T / X.T.sum(axis=1)[:, None]
        else:
            X = X.T / X.T.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.row_masses_.to_numpy() ** -0.5) @ self.svd_.U,
            index=pd.Index(col_names, name=index_name),
        )

    @utils.check_is_dataframe_input
    @select_active_rows
    def column_cosine_similarities(self, X):
        """Return the cos2 for each column against the dimensions.

        The cos2 value gives an indicator of the accuracy of the column projection on the dimension.

        Values above 0.5 usually means that the column is relatively accurately well projected onto that dimension. Its often
        used to identify which factor/dimension is important for a given element as the cos2 can be interpreted as the proportion
        of the variance of the element attributed to a particular factor.
        """
        G = self.column_coordinates(X)
        return self._column_cosine_similarities(X, G)

    @select_active_rows
    def _column_cosine_similarities(self, X, G):
        # Active
        X_act = X[self.active_cols_]
        X_act = X_act / X_act.sum().sum()
        marge_row = X_act.sum(axis=1)
        Tc = X_act.div(marge_row, axis=0).div(X_act.sum(axis=0), axis=1) - 1
        dist2_col = (Tc**2).mul(marge_row, axis=0).sum(axis=0)

        # Supplementary
        X_sup = X[X.columns.difference(self.active_cols_, sort=False)]
        X_sup = X_sup.div(X_sup.sum(axis=0), axis=1)
        dist2_col_sup = ((X_sup.sub(marge_row, axis=0)) ** 2).div(marge_row, axis=0).sum(axis=0)

        dist2_col = pd.concat((dist2_col, dist2_col_sup))
        return (G**2).div(dist2_col, axis=0)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def plot(
        self,
        X,
        x_component=0,
        y_component=1,
        show_row_markers=True,
        show_column_markers=True,
        show_row_labels=False,
        show_column_labels=False,
    ):
        eig = self._eigenvalues_summary.to_dict(orient="index")

        row_chart_markers = None
        row_chart_labels = None
        column_chart_markers = None
        column_chart_labels = None

        if show_row_markers or show_row_labels:
            row_coords = self.row_coordinates(X)
            row_coords.columns = [f"component {i}" for i in row_coords.columns]
            row_coords = row_coords.assign(
                variable=row_coords.index.name or "row",
                value=row_coords.index.astype(str),
            )
            row_labels = pd.Series(row_coords.index, index=row_coords.index)
            row_chart = alt.Chart(row_coords.assign(label=row_labels)).encode(
                x=alt.X(
                    f"component {x_component}",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        title=f"component {x_component} — {eig[x_component]['% of variance'] / 100:.2%}"
                    ),
                ),
                y=alt.Y(
                    f"component {y_component}",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        title=f"component {y_component} — {eig[y_component]['% of variance'] / 100:.2%}"
                    ),
                ),
            )
            row_chart_markers = row_chart.mark_circle(size=50 if show_row_markers else 0).encode(
                color="variable",
                tooltip=[
                    "variable",
                    "value",
                    f"component {x_component}",
                    f"component {y_component}",
                ],
            )
            if show_row_labels:
                row_chart_labels = row_chart.mark_text().encode(text="label:N")

        if show_column_markers or show_column_labels:
            column_coords = self.column_coordinates(X)
            column_coords.columns = [f"component {i}" for i in column_coords.columns]
            column_coords = column_coords.assign(
                variable=column_coords.index.name or "column",
                value=column_coords.index.astype(str),
            )
            column_labels = pd.Series(column_coords.index, index=column_coords.index)
            column_chart = alt.Chart(column_coords.assign(label=column_labels)).encode(
                x=alt.X(
                    f"component {x_component}",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        title=f"component {x_component} — {eig[x_component]['% of variance'] / 100:.2%}"
                    ),
                ),
                y=alt.Y(
                    f"component {y_component}",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        title=f"component {y_component} — {eig[y_component]['% of variance'] / 100:.2%}"
                    ),
                ),
            )
            column_chart_markers = column_chart.mark_circle(
                size=50 if show_column_markers else 0
            ).encode(
                color="variable",
                tooltip=[
                    "variable",
                    "value",
                    f"component {x_component}",
                    f"component {y_component}",
                ],
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
