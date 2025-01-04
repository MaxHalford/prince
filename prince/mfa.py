"""Multiple Factor Analysis (MFA)"""

from __future__ import annotations

import collections

import altair as alt
import numpy as np
import pandas as pd

from prince import pca, utils


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
            rescale_with_mean=True,
            rescale_with_std=True,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
        )
        collections.UserDict.__init__(self)

    @utils.check_is_dataframe_input
    def fit(self, X, y=None, groups=None, supplementary_groups=None):
        # Checks groups are provided
        self.groups_ = self._determine_groups(X, groups)
        if supplementary_groups is not None:
            for group in supplementary_groups:
                if group not in self.groups_:
                    raise ValueError(f"Supplementary group '{group}' is not in the groups")
            self.supplementary_groups_ = supplementary_groups

        # Check group types are consistent
        self.all_nums_ = {}
        for group, cols in sorted(self.groups_.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError(f'Not all columns in "{group}" group are of the same type')
            self.all_nums_[group] = all_num

        # Run a factor analysis in each group
        for group, cols in sorted(self.groups_.items()):
            if self.all_nums_[group]:
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
                raise NotImplementedError("Groups of non-numerical variables are not supported yet")
            self[group] = fa.fit(X.loc[:, cols])

        # Fit the global PCA
        Z = self._build_Z(X)
        column_weights = np.array(
            [
                1 / self[group].eigenvalues_[0]
                for group, cols in self.groups_.items()
                for _ in cols
                if group not in getattr(self, "supplementary_groups_", [])
            ]
        )
        super().fit(
            Z,
            column_weight=column_weights,
            supplementary_columns=[
                column
                for group in getattr(self, "supplementary_groups_", [])
                for column in self.groups_[group]
            ],
        )

        return self

    def _determine_groups(self, X: pd.DataFrame, groups: dict | list | None) -> dict:
        if groups is None:
            if isinstance(X.columns, pd.MultiIndex):
                groups = X.columns.get_level_values(0).unique().tolist()
            else:
                raise ValueError("Groups have to be specified")

        if isinstance(groups, list):
            if not isinstance(X.columns, pd.MultiIndex):
                raise ValueError(
                    "X has to have MultiIndex columns if groups are provided as a list"
                )
            groups = {
                group: [
                    (group, column)
                    for column in X.columns.get_level_values(1)[
                        X.columns.get_level_values(0) == group
                    ]
                ]
                for group in groups
            }
        return groups

    def _build_Z(self, X):
        return pd.concat(
            (X[cols] for _, cols in self.groups_.items()),
            axis="columns",
        )

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        """Returns the row principal coordinates."""
        Z = self._build_Z(X)
        return super().row_coordinates(Z)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def partial_row_coordinates(self, X):
        """Returns the partial row principal coordinates."""
        Z = self._build_Z(X)
        coords = []
        for _, names in self.groups_.items():
            partial_coords = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
            partial_coords.loc[:, names] = (Z[names] - Z[names].mean()) / Z[names].std(ddof=0)
            partial_coords = partial_coords * self.column_weight_
            partial_coords = (len(self.groups_) * partial_coords).dot(self.svd_.V.T)
            coords.append(partial_coords)
        coords = pd.concat(coords, axis=1, keys=self.groups_.keys())
        coords.columns.name = "component"
        return coords

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_coordinates(self, X):
        Z = self._build_Z(X)
        return super().column_coordinates(Z)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def inverse_transform(self, X):
        raise NotImplementedError("MFA inherits from PCA, but this method is not implemented yet")

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_standard_coordinates(self, X):
        Z = self._build_Z(X)
        return super().row_standard_coordinates(Z)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_cosine_similarities(self, X):
        Z = self._build_Z(X)
        return super().row_cosine_similarities(Z)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities_(self, X):
        Z = self._build_Z(X)
        return super().column_cosine_similarities_(Z)

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def plot(self, X, x_component=0, y_component=1, show_partial_rows=False, **params):
        index_name = X.index.name or "index"

        params["tooltip"] = (
            X.index.names if isinstance(X.index, pd.MultiIndex) else [index_name]
        ) + [
            "group",
            f"component {x_component}",
            f"component {y_component}",
        ]

        eig = self._eigenvalues_summary.to_dict(orient="index")

        row_plot = None
        partial_row_plot = None
        edges_plot = None

        # Barycenters
        row_coords = self.row_coordinates(X)
        row_coords.columns = [f"component {i}" for i in row_coords.columns]
        row_coords = row_coords.reset_index()
        row_coords["group"] = "Global"
        if show_partial_rows:
            params["color"] = "group:N"
        row_plot = (
            alt.Chart(row_coords)
            .mark_point(filled=True, size=50)
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

        # Partial row coordinates
        if show_partial_rows:
            partial_row_coords = self.partial_row_coordinates(X).stack(level=0, future_stack=True)
            partial_row_coords.columns = [f"component {i}" for i in partial_row_coords.columns]
            partial_row_coords = partial_row_coords.reset_index(names=[index_name, "group"])

            partial_row_plot = (
                alt.Chart(partial_row_coords)
                .mark_point(shape="circle")
                .encode(
                    alt.X(f"component {x_component}", scale=alt.Scale(zero=False)),
                    alt.Y(f"component {y_component}", scale=alt.Scale(zero=False)),
                    **params,
                )
            )

        # Edges to connect the main markers to the partial markers
        if show_partial_rows:
            edges = pd.merge(
                left=row_coords[
                    [index_name, f"component {x_component}", f"component {y_component}"]
                ],
                right=partial_row_coords[
                    [index_name, f"component {x_component}", f"component {y_component}", "group"]
                ],
                on=index_name,
                suffixes=("_global", "_partial"),
            )
            edges_plot = (
                alt.Chart(edges)
                .mark_line(opacity=0.7)
                .encode(
                    x=f"component {x_component}_global:Q",
                    y=f"component {y_component}_global:Q",
                    x2=f"component {x_component}_partial:Q",
                    y2=f"component {y_component}_partial:Q",
                    color="group:N",
                    strokeDash=alt.value([2, 2]),
                )
            )

        charts = filter(
            None,
            (row_plot, partial_row_plot, edges_plot),
        )

        return alt.layer(*charts).interactive()
