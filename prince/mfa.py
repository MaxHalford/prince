"""Multiple Factor Analysis (MFA)"""
import collections
import itertools

import altair as alt
import numpy as np
import pandas as pd
import sklearn.utils

from prince import mca
from prince import pca
from prince import plot
from prince import utils


class MFA(pca.PCA, collections.UserDict):
    def __init__(
        self,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
    ):
        super().__init__(
            rescale_with_mean=False,
            rescale_with_std=False,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
        )
        collections.UserDict.__init__(self)

    def _check_input(self, X):
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, np.number])

    def fit(self, X, y=None, groups=None):

        # Checks groups are provided
        self.groups_ = self._determine_groups(X, groups)

        # Check input
        self._check_input(X)

        # Check group types are consistent
        self.all_nums_ = {}
        for name, cols in sorted(self.groups_.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError(
                    'Not all columns in "{}" group are of the same type'.format(name)
                )
            self.all_nums_[name] = all_num

        # Normalize column-wise
        X = (X - X.mean()) / ((X - X.mean()) ** 2).sum() ** 0.5

        # Run a factor analysis in each group
        for name, cols in sorted(self.groups_.items()):
            if self.all_nums_[name]:
                fa = pca.PCA(
                    rescale_with_mean=True,
                    rescale_with_std=True,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=True,
                    random_state=self.random_state,
                    engine=self.engine,
                )
            else:
                raise NotImplementedError(
                    "Groups of non-numerical variables are not supported yet"
                )
            self[name] = fa.fit(X.loc[:, cols])

        # Fit the global PCA
        Z = pd.concat(
            (
                X[cols] / self[g].eigenvalues_[0] ** 0.5
                for g, cols in self.groups_.items()
            ),
            axis="columns",
        )
        super().fit(Z)
        self.total_inertia_ = sum(self.eigenvalues_)

        # TODO: column_coordinates_ is not implemented yet
        delattr(self, "column_coordinates_")

        return self

    def _determine_groups(self, X, provided_groups):
        if provided_groups is None:
            raise ValueError("Groups have to be specified")
        if isinstance(provided_groups, list):
            if not isinstance(X.columns, pd.MultiIndex):
                raise ValueError(
                    "Groups have to be provided as a dict when X is not a MultiIndex"
                )
            groups = {
                g: [
                    (g, c)
                    for c in X.columns.get_level_values(1)[
                        X.columns.get_level_values(0) == g
                    ]
                ]
                for g in provided_groups
            }
        else:
            groups = provided_groups
        return groups

    @property
    @utils.check_is_fitted
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        return np.square(self.svd_.s)

    def row_coordinates(self, X):
        """Returns the row principal coordinates."""

        if (X.index != self.row_contributions_.index).any():
            raise NotImplementedError("Supplementary rows are not supported yet")

        X = (X - X.mean()) / ((X - X.mean()) ** 2).sum() ** 0.5
        Z = pd.concat(
            (
                X[cols] / self[g].eigenvalues_[0] ** 0.5
                for g, cols in self.groups_.items()
            ),
            axis="columns",
        )
        U = self.svd_.U
        s = self.svd_.s
        M = np.full(len(X), 1 / len(X))

        return (Z @ Z.T) @ (M[:, np.newaxis] ** (-0.5) * U * s**-1)

    def group_row_coordinates(self, X):

        if (X.index != self.row_contributions_.index).any():
            raise NotImplementedError("Supplementary rows are not supported yet")

        X = (X - X.mean()) / ((X - X.mean()) ** 2).sum() ** 0.5
        Z = pd.concat(
            (
                X[cols] / self[g].eigenvalues_[0] ** 0.5
                for g, cols in self.groups_.items()
            ),
            axis="columns",
        )
        M = np.full(len(X), 1 / len(X))
        U = self.svd_.U
        s = self.svd_.s

        def add_index(g, group_name):
            g.columns = pd.MultiIndex.from_tuples(
                [(group_name, col) for col in g.columns],
                names=("group", "component"),
            )
            return g

        return len(self.groups_) * pd.concat(
            [
                add_index(
                    g=(Z[g] @ Z[g].T) @ (M[:, np.newaxis] ** (-0.5) * U * s**-1),
                    group_name=g,
                )
                for g, cols in self.groups_.items()
            ],
            axis="columns",
        )

    def column_coordinates(self, X):
        raise NotImplemented(
            "MFA inherits from PCA, but this method is not implemented yet"
        )

    def inverse_transform(self, X):
        raise NotImplemented(
            "MFA inherits from PCA, but this method is not implemented yet"
        )

    def row_standard_coordinates(self, X):
        raise NotImplemented(
            "MFA inherits from PCA, but this method is not implemented yet"
        )

    def row_cosine_similarities(self, X):
        raise NotImplemented(
            "MFA inherits from PCA, but this method is not implemented yet"
        )

    def column_correlations(self, X):
        raise NotImplemented(
            "MFA inherits from PCA, but this method is not implemented yet"
        )

    def column_cosine_similarities_(self, X):
        raise NotImplemented(
            "MFA inherits from PCA, but this method is not implemented yet"
        )

    @property
    def column_contributions_(self):
        raise NotImplemented(
            "MFA inherits from PCA, but this method is not implemented yet"
        )

    def plot(self, X, x_component=0, y_component=1, color_by=None, **params):

        if color_by is not None:
            params["color"] = color_by

        params["tooltip"] = (
            X.index.names if isinstance(X.index, pd.MultiIndex) else ["index"]
        ) + [
            f"component {x_component}",
            f"component {y_component}",
        ]

        eig = self._eigenvalues_summary.to_dict(orient="index")

        row_coords = self.row_coordinates(X)
        row_coords.columns = [f"component {i}" for i in row_coords.columns]
        row_coords = row_coords.reset_index()
        row_plot = (
            alt.Chart(row_coords)
            .mark_circle()
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

        return row_plot.interactive()
