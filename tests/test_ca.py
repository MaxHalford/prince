import tempfile
import numpy as np
import pandas as pd
import prince
import rpy2.rinterface_lib
from rpy2.robjects import r as R
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

    def test_row_coords(self):
        F = load_df_from_R("ca$row$coord")
        if self.sup_rows:
            F = pd.concat((F, load_df_from_R("ca$row.sup$coord")))
        P = self.ca.row_coordinates(self.dataset)
        np.testing.assert_allclose(F.abs(), P.abs())


class TestCANoSup(CATestSuite):
    ...


class TestCASupRows(CATestSuite):
    sup_rows = True


class TestCASupCols(CATestSuite):
    sup_cols = True


class TestCASupRowsSupCols(CATestSuite):
    sup_rows = True
    sup_cols = True
