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


class CATestSuite:
    sup_rows = False
    sup_cols = False

    @classmethod
    def setup_class(cls):

        n_components = 5

        # Fit Prince
        cls.dataset = prince.datasets.load_french_elections()
        active = cls.dataset.copy()
        if cls.sup_rows:
            active = active.drop("Île-de-France")
        if cls.sup_cols:
            active = active.drop(columns=["Abstention", "Blank"])
        cls.ca = prince.CA(n_components=n_components)
        cls.ca.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            cls.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}', row.names=1)")

        args = f"dataset, ncp={n_components}, graph=F"
        if cls.sup_cols:
            if cls.sup_rows:
                R(f"ca <- CA({args}, col.sup=c(13, 14), row.sup=c(18))")
            else:
                R(f"ca <- CA({args}, col.sup=c(13, 14))")
        else:
            if cls.sup_rows:
                R(f"ca <- CA({args}, row.sup=c(18))")
            else:
                R(f"ca <- CA({args})")

    def test_check_is_fitted(self):
        assert isinstance(self.ca, prince.CA)
        sklearn.utils.validation.check_is_fitted(self.ca)

    def test_svd_U(self):
        F = load_df_from_R("ca$svd$U").to_numpy()
        P = sparse.diags(self.ca.row_masses_.to_numpy() ** -0.5) @ self.ca.svd_.U
        np.testing.assert_allclose(np.abs(F), np.abs(P))

    def test_svd_V(self):
        F = load_df_from_R("ca$svd$V").to_numpy()
        P = sparse.diags(self.ca.col_masses_.to_numpy() ** -0.5) @ self.ca.svd_.V.T
        np.testing.assert_allclose(np.abs(F), np.abs(P))

    def test_eigenvalues(self):
        F = load_df_from_R("ca$eig")[: self.ca.n_components]
        P = self.ca._eigenvalues_summary
        np.testing.assert_allclose(F["eigenvalue"], P["eigenvalue"])
        np.testing.assert_allclose(F["percentage of variance"], P["% of variance"])
        np.testing.assert_allclose(
            F["cumulative percentage of variance"], P["% of variance (cumulative)"]
        )

    def test_row_coords(self):
        F = load_df_from_R("ca$row$coord")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("ca$row.sup$coord")))
        P = self.ca.row_coordinates(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_row_contrib(self):
        F = load_df_from_R("ca$row$contrib")
        P = self.ca.row_contributions_
        np.testing.assert_allclose(F, P * 100)

    def test_col_coords(self):
        F = load_df_from_R("ca$col$coord")
        if self.sup_cols:
            F = pd.concat((F, load_df_from_R("ca$col.sup$coord")))
        P = self.ca.column_coordinates(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())

    def test_col_contrib(self):
        F = load_df_from_R("ca$col$contrib")
        P = self.ca.column_contributions_
        np.testing.assert_allclose(F, P * 100)


class TestCANoSup(CATestSuite):
    ...


class TestCASupRows(CATestSuite):
    sup_rows = True


class TestCASupCols(CATestSuite):
    sup_cols = True


class TestCASupRowsSupCols(CATestSuite):
    sup_rows = True
    sup_cols = True
