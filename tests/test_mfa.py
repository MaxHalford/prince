from __future__ import annotations

import math
import tempfile

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as robjects
import sklearn.utils.estimator_checks
import sklearn.utils.validation
from rpy2.robjects import r as R

import prince
from tests import load_df_from_R


@pytest.mark.parametrize(
    "sup_rows, sup_groups",
    [
        pytest.param(sup_rows, sup_groups, id=f"{sup_rows=}:{sup_groups=}")
        for sup_rows in [False, True]
        for sup_groups in [False, True]
    ],
)
class TestMFA:
    _row_name = "row"
    _col_name = "col"

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_groups):
        self.sup_rows = sup_rows
        self.sup_groups = sup_groups

        n_components = 3

        # Fit Prince
        self.dataset = prince.datasets.load_premier_league()
        active = self.dataset.copy()
        if self.sup_rows:
            active = active.drop(index=["Manchester City", "Manchester United"])
        supplementary_groups = ["2023-24"] if self.sup_groups else []
        self.groups = self.dataset.columns.levels[0].tolist()
        self.mfa = prince.MFA(n_components=n_components)
        self.mfa.fit(active, groups=self.groups, supplementary_groups=supplementary_groups)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            dataset = self.dataset.copy()
            dataset.columns = [" ".join(parts) for parts in dataset.columns]
            dataset.to_csv(fp, index=False)
            R(f"dataset <- read.csv('{fp.name}')")

        args = "dataset, group=c(6, 6, 6), graph=F"
        if self.sup_rows:
            args += ", ind.sup=c(9:10)"
        if self.sup_groups:
            args += ", num.group.sup=c(3)"

        R(f"mfa <- MFA({args})")

    def test_check_is_fitted(self):
        assert isinstance(self.mfa, prince.MFA)
        sklearn.utils.validation.check_is_fitted(self.mfa)

    def test_total_inertia(self):
        # FactoMineR 2.15 only stores the top `ncp` eigenvalues, so
        # `sum(mfa$eig[,1])` no longer equals the true total inertia.
        # Recover it from the percentage of variance of the first component.
        F = robjects.r("mfa$eig[1,1] / (mfa$eig[1,2] / 100)")[0]
        P = self.mfa.total_inertia_
        assert math.isclose(F, P)

    def test_eigenvalues(self):
        F = load_df_from_R("mfa$eig")[: self.mfa.n_components]
        P = self.mfa._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    def test_group_eigenvalues(self):
        for i, group in enumerate(self.groups, start=1):
            F = load_df_from_R(f"mfa$separate.analyses$Gr{i}$eig")[: self.mfa.n_components]
            P = self.mfa[group]._eigenvalues_summary
            np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
            np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
            np.testing.assert_allclose(
                F["cumulative percentage of variance"], P["% of variance (cumulative)"]
            )

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        method = getattr(self.mfa, method_name)
        F = load_df_from_R("mfa$ind$coord")
        P = method(self.dataset)
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("mfa$ind.sup$coord")))
            # Move supplementary rows to the end
            P = pd.concat(
                [
                    P.loc[P.index.difference(["Manchester City", "Manchester United"])],
                    P.loc[["Manchester City", "Manchester United"]],
                ]
            )
        F = F.iloc[:, : self.mfa.n_components]
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_row_contrib(self):
        F = load_df_from_R("mfa$ind$contrib").iloc[:, : self.mfa.n_components]
        P = self.mfa.row_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_col_coords(self):
        F = load_df_from_R("mfa$quanti.var$coord").iloc[:, : self.mfa.n_components]
        if self.sup_groups:
            F_sup = load_df_from_R("mfa$quanti.var.sup$coord").iloc[:, : self.mfa.n_components]
            F = pd.concat((F, F_sup))
        P = self.mfa.column_coordinates_
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_col_cor(self):
        F = load_df_from_R("mfa$quanti.var$cor").iloc[:, : self.mfa.n_components]
        if self.sup_groups:
            F_sup = load_df_from_R("mfa$quanti.var.sup$cor").iloc[:, : self.mfa.n_components]
            F = pd.concat((F, F_sup))
        P = self.mfa.column_correlations
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_col_cos2(self):
        F = load_df_from_R("mfa$quanti.var$cos2").iloc[:, : self.mfa.n_components]
        if self.sup_groups:
            F_sup = load_df_from_R("mfa$quanti.var.sup$cos2").iloc[:, : self.mfa.n_components]
            F = pd.concat((F, F_sup))
        P = self.mfa.column_cosine_similarities_
        np.testing.assert_allclose(F, P)

    def test_col_contrib(self):
        F = load_df_from_R("mfa$quanti.var$contrib").iloc[:, : self.mfa.n_components]
        P = self.mfa.column_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_group_coords(self):
        F = load_df_from_R("mfa$group$coord").iloc[:, : self.mfa.n_components]
        P = self.mfa.group_coordinates_
        np.testing.assert_allclose(F, P)

    def test_group_contrib(self):
        F = load_df_from_R("mfa$group$contrib").iloc[:, : self.mfa.n_components]
        P = self.mfa.group_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_group_cos2(self):
        F = load_df_from_R("mfa$group$cos2").iloc[:, : self.mfa.n_components]
        P = self.mfa.group_cosine_similarities_
        np.testing.assert_allclose(F, P)

    def test_partial_cor(self):
        F = load_df_from_R("mfa$partial.axes$cor").iloc[:, : self.mfa.n_components]
        P = self.mfa.partial_correlations_
        # FactoMineR always includes all groups (active + supplementary) with ncp=5 per group.
        # Select rows matching prince's active groups and n_components.
        n = self.mfa.n_components
        ncp_facto = F.shape[0] // len(self.groups)
        active_group_indices = (
            [i for i, g in enumerate(self.groups) if g != "2023-24"]
            if self.sup_groups
            else list(range(len(self.groups)))
        )
        indices = [g * ncp_facto + k for g in active_group_indices for k in range(n)]
        np.testing.assert_allclose(F.iloc[indices].abs(), P.abs())

    def test_partial_contrib(self):
        F = load_df_from_R("mfa$partial.axes$contrib").iloc[:, : self.mfa.n_components]
        P = self.mfa.partial_contributions_
        n = self.mfa.n_components
        ncp_facto = F.shape[0] // len(self.groups)
        active_group_indices = (
            [i for i, g in enumerate(self.groups) if g != "2023-24"]
            if self.sup_groups
            else list(range(len(self.groups)))
        )
        indices = [g * ncp_facto + k for g in active_group_indices for k in range(n)]
        # Renormalize contributions since we're comparing a subset of partial axes
        F_subset = F.iloc[indices]
        F_renorm = F_subset / F_subset.sum(axis=0) * 100
        np.testing.assert_allclose(F_renorm, P * 100)

    def test_partial_row_coords(self):
        F = load_df_from_R("mfa$ind$coord.partiel").iloc[:, : self.mfa.n_components]
        if self.sup_rows:
            F_sup = load_df_from_R("mfa$ind.sup$coord.partiel").iloc[:, : self.mfa.n_components]
        P = self.mfa.partial_row_coordinates(self.dataset)

        active_groups = [g for g in self.groups if not self.sup_groups or g != "2023-24"]
        n_active_groups = len(active_groups)
        sup_rows = ["Manchester City", "Manchester United"]

        for i, group in enumerate(active_groups):
            F_group = F.iloc[i::n_active_groups]
            P_group = P[group]
            if self.sup_rows:
                P_group_active = P_group.drop(sup_rows)
                np.testing.assert_allclose(F_group.abs().values, P_group_active.abs().values)
                F_sup_group = F_sup.iloc[i::n_active_groups]
                P_sup_group = P_group.loc[sup_rows]
                np.testing.assert_allclose(F_sup_group.abs().values, P_sup_group.abs().values)
            else:
                np.testing.assert_allclose(F_group.abs().values, P_group.abs().values)


def test_mfa_non_numeric_supports_categorical():
    """A group made of string columns should fit (no longer raises NotImplementedError)."""
    mfa = prince.MFA(n_components=3).fit(prince.datasets.load_poison())
    sklearn.utils.validation.check_is_fitted(mfa)


@pytest.mark.parametrize(
    "sup_rows, sup_groups",
    [
        pytest.param(sup_rows, sup_groups, id=f"{sup_rows=}:{sup_groups=}")
        for sup_rows in [False, True]
        for sup_groups in [False, True]
    ],
)
class TestMFACategorical:
    """MFA on the FactoMineR ``poison`` dataset with one numerical and three categorical groups.

    Exercises mixed numeric/categorical MFA (FactoMineR's ``type=c('s','n','n','n')``)
    under all combinations of supplementary rows and a supplementary (categorical) group.
    The supplementary group chosen is ``eat`` because it has the most categories, which
    stresses the indicator-block code path.
    """

    _sup_group_name = "foods"
    # Rows 0 and 1 keep every categorical level represented in the remaining 53 rows,
    # which is what FactoMineR requires for ind.sup with type 'n' groups.
    _sup_row_indices = [0, 1]

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_groups):
        self.sup_rows = sup_rows
        self.sup_groups = sup_groups
        self.n_components = 3

        self.dataset = prince.datasets.load_poison()
        # Group order matches FactoMineR's poison: description, illness, symptoms, foods.
        self.group_names = self.dataset.columns.get_level_values(0).unique().tolist()

        active = self.dataset.copy()
        if self.sup_rows:
            active = active.drop(index=self._sup_row_indices)
        supplementary_groups = [self._sup_group_name] if self.sup_groups else []
        # engine="scipy" forces a deterministic full SVD so the tight (atol=1e-4)
        # comparisons against FactoMineR don't flake on randomized-SVD precision noise
        # on the last component (see CI flake on PR #236).
        self.mfa = prince.MFA(n_components=self.n_components, engine="scipy").fit(
            active,
            supplementary_groups=supplementary_groups,
        )

        R("library('FactoMineR')")
        R("data(poison)")
        type_vec = "c('s','n','n','n')"
        group_sizes = "c(2,2,5,6)"
        name_group = "c('" + "','".join(self.group_names) + "')"
        args = (
            f"poison, group={group_sizes}, type={type_vec}, "
            f"name.group={name_group}, ncp={self.n_components}, graph=F"
        )
        if self.sup_rows:
            # FactoMineR uses 1-based indices.
            args += f", ind.sup=c({','.join(str(i + 1) for i in self._sup_row_indices)})"
        if self.sup_groups:
            sup_idx = self.group_names.index(self._sup_group_name) + 1
            args += f", num.group.sup=c({sup_idx})"
        R(f"mfa <- MFA({args})")

    def _active_group_names(self):
        return [g for g in self.group_names if not (self.sup_groups and g == self._sup_group_name)]

    def test_check_is_fitted(self):
        sklearn.utils.validation.check_is_fitted(self.mfa)

    def test_total_inertia(self):
        # FactoMineR 2.15 only stores the top `ncp` eigenvalues, so
        # `sum(mfa$eig[,1])` no longer equals the true total inertia.
        # Recover it from the percentage of variance of the first component.
        F = robjects.r("mfa$eig[1,1] / (mfa$eig[1,2] / 100)")[0]
        assert math.isclose(F, self.mfa.total_inertia_)

    def test_eigenvalues(self):
        F = load_df_from_R("mfa$eig")[: self.n_components]
        P = self.mfa._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        method = getattr(self.mfa, method_name)
        F = load_df_from_R("mfa$ind$coord")
        P = method(self.dataset)
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("mfa$ind.sup$coord")))
            P = pd.concat(
                [
                    P.loc[P.index.difference(self._sup_row_indices)],
                    P.loc[self._sup_row_indices],
                ]
            )
        F = F.iloc[:, : self.n_components]
        np.testing.assert_allclose(F.abs().values, P.abs().values, atol=1e-4)

    def test_row_contrib(self):
        F = load_df_from_R("mfa$ind$contrib").iloc[:, : self.n_components]
        P = self.mfa.row_contributions_
        np.testing.assert_allclose(F.values, P.values * 100, atol=1e-3)

    def test_group_coords(self):
        F = load_df_from_R("mfa$group$coord").iloc[:, : self.n_components]
        P = self.mfa.group_coordinates_
        np.testing.assert_allclose(F.values, P.values, atol=1e-4)

    def test_group_contrib(self):
        F = load_df_from_R("mfa$group$contrib").iloc[:, : self.n_components]
        P = self.mfa.group_contributions_
        np.testing.assert_allclose(F.values, P.values * 100, atol=1e-3)

    def test_group_cos2(self):
        F = load_df_from_R("mfa$group$cos2").iloc[:, : self.n_components]
        P = self.mfa.group_cosine_similarities_
        np.testing.assert_allclose(F.values, P.values, atol=1e-4)

    def test_partial_row_coords(self):
        F = load_df_from_R("mfa$ind$coord.partiel").iloc[:, : self.n_components]
        if self.sup_rows:
            F_sup = load_df_from_R("mfa$ind.sup$coord.partiel").iloc[:, : self.n_components]
        P = self.mfa.partial_row_coordinates(self.dataset)
        active_groups = self._active_group_names()
        n_active_groups = len(active_groups)
        for i, group in enumerate(active_groups):
            F_group = F.iloc[i::n_active_groups]
            P_group = P[group]
            if self.sup_rows:
                P_group_active = P_group.drop(self._sup_row_indices)
                np.testing.assert_allclose(
                    F_group.abs().values, P_group_active.abs().values, atol=1e-4
                )
                F_sup_group = F_sup.iloc[i::n_active_groups]
                P_sup_group = P_group.loc[self._sup_row_indices]
                np.testing.assert_allclose(
                    F_sup_group.abs().values, P_sup_group.abs().values, atol=1e-4
                )
            else:
                np.testing.assert_allclose(F_group.abs().values, P_group.abs().values, atol=1e-4)

    def test_quanti_var_coords(self):
        """Column coordinates for the numerical variables should match FactoMineR's quanti.var$coord."""
        F = load_df_from_R("mfa$quanti.var$coord").iloc[:, : self.n_components]
        num_cols = [c for c in self.dataset.columns if c[0] == "description"]
        P = self.mfa.column_coordinates_.loc[num_cols]
        np.testing.assert_allclose(F.abs().values, P.abs().values, atol=1e-4)

    def test_partial_axis_correlations(self):
        """Partial-axis correlations should match FactoMineR for the active groups.

        FactoMineR's ``mfa$partial.axes$cor`` includes both active and supplementary
        groups; we filter rows to active groups only and verify shape + values.
        """
        F = load_df_from_R("mfa$partial.axes$cor").iloc[:, : self.n_components]
        # FactoMineR row labels look like "Dim<k>.<group>"; keep only active groups.
        active = set(self._active_group_names())
        F = F.loc[[lbl for lbl in F.index if lbl.split(".", 1)[1] in active]]
        P = self.mfa.partial_correlations_
        assert F.shape == P.shape
        np.testing.assert_allclose(F.abs().values, P.abs().values, atol=1e-4)
