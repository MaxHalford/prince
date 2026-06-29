"""Multiple Factor Analysis (MFA)"""

from __future__ import annotations

import collections
import enum
from typing import Any, cast

import altair as alt
import numpy as np
import pandas as pd
from typing_extensions import override

from prince import mca, pca, utils


class GroupType(enum.Enum):
    """Type of variables in an MFA group.

    A group must be homogeneous: every column in it is either numerical (PCA-style)
    or categorical (MCA-style). Mixed groups are not supported.
    """

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


# A group's name and the list of (output) column names it maps to. The per-group
# preprocessing dict mixes value types (enums, ndarrays, name lists, ints), so it is
# typed loosely as ``dict[str, Any]`` and concrete element types are recovered via casts
# at the use sites below.
Preprocessing = dict[str, Any]


class MFA(pca.PCA, collections.UserDict[Any, Any]):
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

    @override
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
        self.group_types_ = {}
        for group, cols in sorted(self.groups_.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(
                pd.api.types.is_string_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)
                for c in cols
            )
            if not (all_num or all_cat):
                raise ValueError(f'Not all columns in "{group}" group are of the same type')
            self.group_types_[group] = GroupType.NUMERICAL if all_num else GroupType.CATEGORICAL

        # Fit a factor analysis in each group and record per-group preprocessing.
        # Numeric groups use PCA (centering / standardization controlled by self.rescale_*).
        # Categorical groups use MCA on an indicator matrix; the corresponding block of Z is
        # built by centering each indicator column and dividing by sqrt(p_j), as in FactoMineR.
        self._group_preprocessing_: dict[Any, Preprocessing] = {}
        for group, cols in sorted(self.groups_.items()):
            X_g = X.loc[:, cols]
            if self.group_types_[group] is GroupType.NUMERICAL:
                fa = pca.PCA(
                    rescale_with_mean=self.rescale_with_mean,
                    rescale_with_std=self.rescale_with_std,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=True,
                    random_state=self.random_state,
                    engine=self.engine,
                ).fit(X_g)
                if hasattr(fa, "scaler_"):
                    mean = fa.scaler_.mean_ if self.rescale_with_mean else np.zeros(len(cols))
                    scale = fa.scaler_.scale_ if self.rescale_with_std else np.ones(len(cols))
                else:
                    mean = np.zeros(len(cols))
                    scale = np.ones(len(cols))
                self._group_preprocessing_[group] = {
                    "type": GroupType.NUMERICAL,
                    "cols": list(cols),
                    "mean": np.asarray(mean, dtype=np.float64),
                    "scale": np.asarray(scale, dtype=np.float64),
                    "output_names": list(cols),
                }
            else:
                fa = mca.MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=True,
                    random_state=self.random_state,
                    engine=self.engine,
                ).fit(X_g)
                indicator = pd.get_dummies(
                    X_g, columns=list(X_g.columns), prefix_sep="__", dtype=float
                )
                prop = indicator.mean(axis=0).to_numpy(dtype=np.float64)
                # FactoMineR's "type='n'" normalization: each indicator column is centered
                # and standardized (dividing by sqrt(p_j*(1-p_j))) so each column has variance 1.
                scale = np.sqrt(prop * (1 - prop))
                scale[scale <= 1e-12] = 1.0
                self._group_preprocessing_[group] = {
                    "type": GroupType.CATEGORICAL,
                    "cols": list(cols),
                    "indicator_columns": list(indicator.columns),
                    "prop": prop,
                    "mean": prop,
                    "scale": scale,
                    "n_vars": len(cols),
                    "output_names": list(indicator.columns),
                }
            self[group] = fa

        # Build the pre-scaled global Z. Each block is already centered and normalized so
        # the global PCA below runs with no further scaling.
        Z = self._build_Z(X)

        sup_groups = getattr(self, "supplementary_groups_", [])
        sup_columns = [
            name for g in sup_groups for name in self._group_preprocessing_[g]["output_names"]
        ]

        # Column weights make each group contribute the same first-eigenvalue inertia (1).
        # For a numerical group this is 1/lambda_1 per column (FactoMineR type "s").
        # For a categorical group it is (1-p_j)/(lambda_1*Q) per indicator column
        # (FactoMineR type "n"): with the variance-1 normalization above, that yields
        # group block inertia equal to MCA_total_inertia / lambda_1.
        active_blocks = []
        for group in self.groups_:
            if group in sup_groups:
                continue
            preprocessing = self._group_preprocessing_[group]
            lambda_1 = self[group].eigenvalues_[0]
            if preprocessing["type"] is GroupType.NUMERICAL:
                active_blocks.append(np.full(len(preprocessing["output_names"]), 1.0 / lambda_1))
            else:
                active_blocks.append(
                    (1 - preprocessing["prop"]) / (lambda_1 * preprocessing["n_vars"])
                )
        column_weights = np.concatenate(active_blocks) if active_blocks else np.array([])

        # Compute group squared distances (Lg(group, group)) using the pre-scaled block
        # and the column weights, matching FactoMineR's funcLg formula:
        #   Lg = sum_{j,k} cov(Z_j, Z_k)^2 * w_j * w_k
        # For numerical groups the column weight is constant 1/lambda_1, recovering the
        # trace(S^2)/lambda_1^2 formula. For categorical groups the per-column weight
        # (1-p_j)/(lambda_1*Q) varies, so we apply it explicitly.
        self._group_dist2_ = {}
        n = len(X)
        offset = 0
        for group in self.groups_:
            if group in sup_groups:
                continue
            block_cols = self._group_preprocessing_[group]["output_names"]
            Z_g = Z.loc[:, block_cols].to_numpy(dtype=np.float64)
            S = Z_g.T @ Z_g / n
            w = column_weights[offset : offset + len(block_cols)]
            # sum_{j,k} S_jk^2 w_j w_k is the quadratic form w' (S∘S) w; the BLAS
            # matvec route is several times faster than materialising S**2 and outer(w, w).
            self._group_dist2_[group] = float(w @ (S * S) @ w)
            offset += len(block_cols)

        # The global PCA sees an already-scaled Z, so disable its scaler. We restore the
        # user-provided flags afterward so they remain inspectable on the fitted instance.
        prev_rescale_mean = self.rescale_with_mean
        prev_rescale_std = self.rescale_with_std
        self.rescale_with_mean = False
        self.rescale_with_std = False
        try:
            super().fit(
                Z,
                column_weight=column_weights,
                supplementary_columns=sup_columns,
            )
        finally:
            self.rescale_with_mean = prev_rescale_mean
            self.rescale_with_std = prev_rescale_std

        # Precompute integer column positions for fast slicing in transform methods.
        # Z is built group-by-group in self.groups_ order, so this matches Z's layout.
        all_z_cols = [
            name
            for group in self.groups_
            for name in self._group_preprocessing_[group]["output_names"]
        ]
        z_col_map = {c: i for i, c in enumerate(all_z_cols)}
        self._z_col_indices_ = np.array([z_col_map[c] for c in Z.columns])

        active_col_map = {c: i for i, c in enumerate(self.feature_names_in_)}
        self._group_col_indices_ = {}
        for group in self._active_groups():
            names = self._group_preprocessing_[group]["output_names"]
            self._group_col_indices_[group] = np.array([active_col_map[n] for n in names])

        return self

    def _determine_groups(
        self, X: pd.DataFrame, groups: dict[Any, Any] | list[Any] | None
    ) -> dict[Any, Any]:
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
        """Build the global pre-scaled Z block by applying each group's preprocessing.

        Active groups come first and supplementary groups are appended at the end,
        so downstream slicing of the active block via ``Z[:, :n_active]`` is correct
        even when the user interleaves a supplementary group between active groups.
        """
        sup_groups = getattr(self, "supplementary_groups_", [])
        active = [g for g in self.groups_ if g not in sup_groups]
        sup = [g for g in self.groups_ if g in sup_groups]
        blocks = [self._scale_group(X, group) for group in active + sup]
        return pd.concat(blocks, axis="columns")

    def _scale_group(self, X, group):
        """Apply the fitted per-group preprocessing to produce that group's Z block."""
        preprocessing = self._group_preprocessing_[group]
        X_g = X.loc[:, preprocessing["cols"]]
        if preprocessing["type"] is GroupType.NUMERICAL:
            arr = (X_g.to_numpy(dtype=np.float64) - preprocessing["mean"]) / preprocessing["scale"]
        else:
            indicator = pd.get_dummies(
                X_g.astype(str), columns=list(X_g.columns), prefix_sep="__", dtype=float
            ).reindex(columns=preprocessing["indicator_columns"], fill_value=0.0)
            arr = (indicator.to_numpy(dtype=np.float64) - preprocessing["mean"]) / preprocessing[
                "scale"
            ]
        return pd.DataFrame(arr, index=X.index, columns=preprocessing["output_names"])

    def _extract_Z_numpy(self, X):
        """Extract the pre-scaled Z as a numpy array, in active-then-supplementary order."""
        return self._build_Z(X).to_numpy(dtype=np.float64)

    def _scale_active_numpy(self, Z_np):
        """Return the active columns of Z. Z is already pre-scaled, so this is a slice."""
        n_active = len(self.feature_names_in_)
        return Z_np[:, :n_active]

    @override
    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        """Returns the row principal coordinates."""
        Z_np = self._extract_Z_numpy(X)
        Z_scaled = self._scale_active_numpy(Z_np)
        Z_scaled *= self.column_weight_
        coord = pd.DataFrame(
            data=Z_scaled @ self.svd_.V.T,
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )
        coord.columns.name = "component"
        return coord

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def partial_row_coordinates(self, X):
        """Returns the partial row principal coordinates."""
        Z_np = self._extract_Z_numpy(X)
        Z_scaled = self._scale_active_numpy(Z_np)
        active_groups = self._active_groups()
        K = len(active_groups)
        V = self.svd_.V
        weights = self.column_weight_

        # For each group, only the group's columns are non-zero, so we can
        # skip the full zero-array and do a smaller matrix multiply:
        # (K * Z_group * w_group) @ V[:, group_cols].T
        coords = []
        for group in active_groups:
            col_idx = self._group_col_indices_[group]
            group_data = Z_scaled[:, col_idx] * (K * weights[col_idx])
            coords.append(group_data @ V[:, col_idx].T)

        result = pd.concat(
            [pd.DataFrame(c, index=X.index) for c in coords],
            axis=1,
            keys=active_groups.keys(),
        )
        result.columns.name = "component"
        return result

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_coordinates(self, X):
        Z = self._build_Z(X)
        # PCA does not define ``column_coordinates`` in its own MRO, but the concrete
        # group factor analyses do; cast to ``Any`` so the dynamic dispatch type-checks.
        return cast(Any, super()).column_coordinates(Z)

    @override
    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def inverse_transform(self, X):
        raise NotImplementedError("MFA inherits from PCA, but this method is not implemented yet")

    @override
    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_standard_coordinates(self, X):
        return self.row_coordinates(X).div(self.eigenvalues_, axis="columns")

    @override
    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_cosine_similarities(self, X):
        Z_np = self._extract_Z_numpy(X)
        Z_scaled = self._scale_active_numpy(Z_np)
        squared_coordinates = (Z_scaled * Z_scaled) @ self.column_weight_
        Z_scaled *= self.column_weight_
        coord = Z_scaled @ self.svd_.V.T
        return pd.DataFrame(
            data=coord**2 / squared_coordinates[:, np.newaxis],
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )

    def _active_groups(self):
        """Mapping of active group name → list of its columns in Z (output names).

        For numerical groups the output names are the original column names; for
        categorical groups they are the indicator (one-hot) column names.
        """
        supplementary_groups = getattr(self, "supplementary_groups_", [])
        return {
            group: self._group_preprocessing_[group]["output_names"]
            for group in self.groups_
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
        n_comp = self.n_components

        # Categorical groups have at most (J - Q) non-trivial dimensions; the SVD may
        # still return additional near-zero eigenvalues. We emit a row per dimension
        # whose eigenvalue is non-trivial (matches FactoMineR), rather than padding
        # shorter groups with phantom rows.
        eig_trivial = 1e-12
        blocks = []
        labels = []
        eig_ratios_list = []
        for group in active_groups:
            group_pca = self[group]
            group_eigs = group_pca.eigenvalues_
            k_avail = min(
                n_comp,
                group_pca.svd_.U.shape[1],
                int(np.sum(group_eigs > group_eigs[0] * eig_trivial)),
            )
            if k_avail == 0:
                continue
            blocks.append(group_pca.svd_.U[:, :k_avail] * group_eigs[:k_avail] ** 0.5)
            for k in range(k_avail):
                labels.append((group, k))
                eig_ratios_list.append(group_eigs[k] / group_eigs[0])
        tab = np.concatenate(blocks, axis=1) if blocks else np.zeros((n_active, 0))
        eig_ratios = np.array(eig_ratios_list)

        # Center and standardize with row weights
        weighted_mean = row_weights @ tab
        tab = tab - weighted_mean
        sigma = np.sqrt(row_weights @ (tab * tab))
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

    @utils.check_is_fitted
    def plot_partial(self, x_component=0, y_component=1):
        """Plot the partial axes correlations.

        Each arrow represents a group's PCA component projected onto the global MFA space.
        Arrows close to the unit circle indicate a strong relationship between the group's
        axis and the global axis.

        """
        cor = self.partial_correlations_
        eig = self._eigenvalues_summary.to_dict(orient="index")

        x_col = f"component {x_component}"
        y_col = f"component {y_component}"

        df = cor[[x_component, y_component]].copy()
        df.columns = [x_col, y_col]
        df = df.reset_index()
        df["group"] = df["group"].astype(str)
        df["label"] = df["group"] + " dim " + df["component"].astype(str)
        df["origin_x"] = 0.0
        df["origin_y"] = 0.0

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_df = pd.DataFrame({x_col: np.cos(theta), y_col: np.sin(theta)})

        x_axis = alt.X(
            x_col,
            scale=alt.Scale(domain=[-1.1, 1.1]),
            axis=alt.Axis(
                title=f"component {x_component} — {eig[x_component]['% of variance'] / 100:.2%}"
            ),
        )
        y_axis = alt.Y(
            y_col,
            scale=alt.Scale(domain=[-1.1, 1.1]),
            axis=alt.Axis(
                title=f"component {y_component} — {eig[y_component]['% of variance'] / 100:.2%}"
            ),
        )

        circle = (
            alt.Chart(circle_df)
            .mark_line(color="lightgray", strokeDash=[4, 4])
            .encode(x=x_axis, y=y_axis, order="index:O")
        )

        arrows = (
            alt.Chart(df)
            .mark_rule()
            .encode(
                x=alt.X("origin_x:Q", scale=alt.Scale(domain=[-1.1, 1.1])),
                y=alt.Y("origin_y:Q", scale=alt.Scale(domain=[-1.1, 1.1])),
                x2=alt.X2(f"{x_col}:Q"),
                y2=alt.Y2(f"{y_col}:Q"),
                color=alt.Color("group:N"),
                tooltip=["label", x_col, y_col],
            )
        )

        points = (
            alt.Chart(df)
            .mark_point(filled=True, size=40)
            .encode(
                x=x_axis,
                y=y_axis,
                color=alt.Color("group:N"),
                tooltip=["label", x_col, y_col],
            )
        )

        labels = (
            alt.Chart(df)
            .mark_text(dx=7, dy=-7, align="left")
            .encode(
                x=x_axis,
                y=y_axis,
                text="label:N",
                color=alt.Color("group:N"),
            )
        )

        return (
            alt.layer(circle, arrows, points, labels)
            .properties(width=400, height=400)
            .interactive()
        )

    @override
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
        partial_row_coords: pd.DataFrame | None = None

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
            # ``partial_row_coords`` is always assigned in the preceding
            # ``if show_partial_rows`` block; narrow it away from ``None``.
            assert partial_row_coords is not None
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
