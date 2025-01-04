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
        F = robjects.r("sum(mfa$eig[,1])")[0]
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
