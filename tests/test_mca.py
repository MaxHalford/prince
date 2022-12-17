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
from tests.test_ca import CATestSuite


class MCATestSuite(CATestSuite):
    _row_name = "ind"
    _col_name = "var"
    sup_rows = False
    sup_cols = False

    @classmethod
    def setup_class(cls):

        n_components = 5

        # Fit Prince
        cls.dataset = prince.datasets.load_hearthstone_cards()
        active = cls.dataset.copy()
        # if cls.sup_rows:
        #     active = active.drop("Île-de-France")
        # if cls.sup_cols:
        #     active = active.drop(columns=["Abstention", "Blank"])
        cls.ca = prince.MCA(n_components=n_components, engine="scipy")
        cls.ca.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            cls.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}')[,-1]")

        args = f"dataset, ncp={n_components}, graph=F"
        R(f"ca <- MCA({args})")

        # if cls.sup_cols:
        #     if cls.sup_rows:
        #         R(f"ca <- CA({args}, col.sup=c(13, 14), row.sup=c(18))")
        #     else:
        #         R(f"ca <- CA({args}, col.sup=c(13, 14))")
        # else:
        #     if cls.sup_rows:
        #         R(f"ca <- CA({args}, row.sup=c(18))")
        #     else:
        #         R(f"ca <- CA({args})")


class TestMCANoSup(MCATestSuite):
    ...
