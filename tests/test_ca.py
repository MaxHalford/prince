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

        # Fit Prince
        cls.dataset = prince.datasets.load_french_elections()
        active = cls.dataset.copy()
        cls.ca = prince.CA()
        cls.ca.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            cls.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}', row.names=1)")

        R(f"ca <- CA(dataset, graph=F)")

    def test_whatever(self):
        assert True


class TestCANoSup(CATestSuite):
    ...


class TestCASupRows(CATestSuite):
    sup_rows = True


class TestCASupCols(CATestSuite):
    sup_cols = True


class TestCASupRowsSupCols(CATestSuite):
    sup_rows = True
    sup_cols = True
