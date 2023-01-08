import tempfile
import numpy as np
import pandas as pd
import prince
import rpy2.rinterface_lib
from rpy2.robjects import r as R
from scipy import sparse
import sklearn.utils.estimator_checks
import sklearn.utils.validation

from tests import load_df_from_R


class MFATestSuite:
    _row_name = "row"
    _col_name = "col"
    sup_rows = False
    sup_cols = False

    @classmethod
    def setup_class(cls):

        n_components = 5

        # Fit Prince
        cls.dataset = prince.datasets.load_burgundy_wines()
        active = cls.dataset.copy()
        # if cls.sup_rows:
        #     active = active.drop("Île-de-France")
        # if cls.sup_cols:
        #     active = active.drop(columns=["Abstention", "Blank"])
        cls.groups = cls.dataset.columns.levels[0].drop("Oak type").tolist()
        cls.mfa = prince.MFA(n_components=n_components)
        cls.mfa.fit(active, groups=cls.groups)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            dataset = cls.dataset.copy()
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


class TestMFANoSup(MFATestSuite):
    ...
