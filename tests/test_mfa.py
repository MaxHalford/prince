import tempfile
import numpy as np
import pandas as pd
import prince
import pytest
import rpy2.rinterface_lib
from rpy2.robjects import r as R
from scipy import sparse
import sklearn.utils.estimator_checks
import sklearn.utils.validation

from tests import load_df_from_R


@pytest.mark.parametrize(
    "sup_rows, sup_cols",
    [
        pytest.param(
            sup_rows,
            sup_cols,
            id=":".join(
                ["sup_rows" if sup_rows else "", "sup_cols" if sup_cols else ""]
            ).strip(":"),
        )
        for sup_rows in [False]
        for sup_cols in [False]
    ],
)
class TestMFA:
    _row_name = "row"
    _col_name = "col"

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_cols):

        self.sup_rows = sup_rows
        self.sup_cols = sup_cols

        n_components = 5

        # Fit Prince
        self.dataset = prince.datasets.load_burgundy_wines()
        active = self.dataset.copy()
        # if self.sup_rows:
        #     active = active.drop("Île-de-France")
        # if self.sup_cols:
        #     active = active.drop(columns=["Abstention", "Blank"])
        self.groups = self.dataset.columns.levels[0].drop("Oak type").tolist()
        self.mfa = prince.MFA(n_components=n_components)
        self.mfa.fit(active, groups=self.groups)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            dataset = self.dataset.copy()
            dataset.columns = [" ".join(parts) for parts in dataset.columns]
            dataset.to_csv(fp, index=False)
            R(f"dataset <- read.csv('{fp.name}')")
            R(f"dataset <- dataset[,-1]")
            R(f"mfa <- MFA(dataset, group=c(3, 4, 3), graph=F)")

    def test_check_is_fitted(self):
        assert isinstance(self.mfa, prince.MFA)
        sklearn.utils.validation.check_is_fitted(self.mfa)

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
            F = load_df_from_R(f"mfa$separate.analyses$Gr{i}$eig")[
                : self.mfa.n_components
            ]
            P = self.mfa[group]._eigenvalues_summary
            np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
            np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
            np.testing.assert_allclose(
                F["cumulative percentage of variance"], P["% of variance (cumulative)"]
            )

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        method = getattr(self.mfa, method_name)
        F = load_df_from_R(f"mfa$ind$coord")
        P = method(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_row_contrib(self):
        F = load_df_from_R("mfa$ind$contrib")
        P = self.mfa.row_contributions_
        np.testing.assert_allclose(F, P * 100)
