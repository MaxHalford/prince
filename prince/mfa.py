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
        rescale_with_mean=True,
        rescale_with_std=True,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
    ):
        super().__init__(
            rescale_with_mean=rescale_with_mean,
            rescale_with_std=rescale_with_std,
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
                    rescale_with_mean=self.rescale_with_mean,
                    rescale_with_std=self.rescale_with_std,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=True,
                    random_state=self.random_state,
                    engine=self.engine,
                )
            else:
                raise NotImplementedError("Groups of non-numerical variables are not supported yet")
            self[group] = fa.fit(X.loc[:, cols])

        # Compute group squared distances (Lg coefficient) using all eigenvalues.
        # trace(S^2) / eigenvalue_1^2, where S is the weighted covariance of the group data.
        self._group_dist2_ = {}
        n = len(X)
        for group, cols in self.groups_.items():
            if group in (supplementary_groups or []):
                continue
            group_pca = self[group]
            X_g = X.loc[:, cols].to_numpy(dtype=np.float64)
            if self.rescale_with_mean or self.rescale_with_std:
                X_g = group_pca.scaler_.transform(X_g)
            S = X_g.T @ X_g / n
            self._group_dist2_[group] = np.sum(S**2) / group_pca.eigenvalues_[0] ** 2

        # Fit the global PCA
        Z = self._build_Z(X)
        sup_groups = getattr(self, "supplementary_groups_", [])
        column_weights = np.array(
            [
                1 / self[group].eigenvalues_[0]
                for group, cols in self.groups_.items()
                for _ in cols
                if group not in sup_groups
            ]
        )
        super().fit(
            Z,
            column_weight=column_weights,
            supplementary_columns=[
                column
                for group in sup_groups
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
        active_groups = self._active_groups()
        active_columns = self.feature_names_in_

        # Scale active columns using fitted scaler parameters
        Z_active = Z[active_columns]
        Z_scaled = pd.DataFrame(
            self.scaler_.transform(Z_active.to_numpy()),
            index=Z.index,
            columns=active_columns,
        )

        coords = []
        for _, names in active_groups.items():
            partial_coords = pd.DataFrame(0.0, index=Z.index, columns=active_columns)
            partial_coords.loc[:, names] = Z_scaled[names]
            partial_coords = partial_coords * self.column_weight_
            partial_coords = (len(active_groups) * partial_coords).dot(self.svd_.V.T)
            coords.append(partial_coords)
        coords = pd.concat(coords, axis=1, keys=active_groups.keys())
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

    def _active_groups(self):
        supplementary_groups = getattr(self, "supplementary_groups_", [])
        return {
            group: names
            for group, names in self.groups_.items()
            if group not in supplementary_groups
        }

    @property
    @utils.check_is_fitted
    def group_contributions_(self):
        """Returns the contribution of each group to each component.

        This is the sum of the variable contributions for all variables in the group.

        """
        col_contrib = self.column_contributions_
        result = {}
        for group, names in self._active_groups().items():
            result[group] = col_contrib.loc[names].sum(axis=0)
        df = pd.DataFrame(result).T
        df.index.name = "group"
        df.columns.name = "component"
        return df

    @property
    @utils.check_is_fitted
    def group_coordinates_(self):
        """Returns the coordinates of each group on each component.

        This is the group contribution scaled by the eigenvalue.

        """
        return self.group_contributions_ * self.eigenvalues_

    @property
    @utils.check_is_fitted
    def group_cosine_similarities_(self):
        """Returns the quality of representation of each group on each component."""
        coord = self.group_coordinates_
        dist2 = pd.Series(self._group_dist2_)
        return (coord**2).div(dist2, axis=0)

    def _partial_axes_table(self):
        """Build the standardized partial axes table and project onto global MFA axes.

        Returns (coord, labels, eig_ratios) where coord is the projected coordinates,
        labels are (group, component) tuples, and eig_ratios are eigenvalue ratios per axis.

        """
        active_groups = self._active_groups()
        n_active = len(self.svd_.U)
        row_weights = np.full(n_active, 1.0 / n_active)

        partial = []
        labels = []
        eig_ratios = []
        for group in active_groups:
            group_pca = self[group]
            row_coords = group_pca.svd_.U * group_pca.eigenvalues_**0.5
            for k in range(self.n_components):
                partial.append(row_coords[:, k])
                labels.append((group, k))
                eig_ratios.append(group_pca.eigenvalues_[k] / group_pca.eigenvalues_[0])

        tab = np.column_stack(partial)

        # Center and standardize with row weights
        weighted_mean = (tab * row_weights[:, np.newaxis]).sum(axis=0)
        tab = tab - weighted_mean
        sigma = np.sqrt((tab**2 * row_weights[:, np.newaxis]).sum(axis=0))
        sigma[sigma < 1e-08] = 1
        tab = tab / sigma

        coord = (tab * row_weights[:, np.newaxis]).T @ self.svd_.U[:, : self.n_components]
        return coord, labels, np.array(eig_ratios)

    @property
    @utils.check_is_fitted
    def partial_correlations_(self):
        """Returns the correlations between each group's PCA axes and the global MFA axes."""
        coord, labels, _ = self._partial_axes_table()
        index = pd.MultiIndex.from_tuples(labels, names=["group", "component"])
        df = pd.DataFrame(coord, index=index)
        df.columns.name = "component"
        return df

    @property
    @utils.check_is_fitted
    def partial_contributions_(self):
        """Returns the contribution of each group's PCA axes to each global MFA component."""
        coord, labels, eig_ratios = self._partial_axes_table()
        raw = coord**2 * eig_ratios[:, np.newaxis]
        contrib = raw / raw.sum(axis=0, keepdims=True)
        index = pd.MultiIndex.from_tuples(labels, names=["group", "component"])
        df = pd.DataFrame(contrib, index=index)
        df.columns.name = "component"
        return df

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
