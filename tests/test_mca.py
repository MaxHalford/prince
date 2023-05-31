from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import r as R

import prince
from tests import load_df_from_R
from tests.test_ca import TestCA as _TestCA


class TestMCA(_TestCA):
    _row_name = "ind"
    _col_name = "var"

    @pytest.fixture(autouse=True)
    def _prepare(self, sup_rows, sup_cols):

        self.sup_rows = sup_rows
        self.sup_cols = sup_cols

        n_components = 5
        n_active_rows = 1_000

        # Fit Prince
        self.dataset = prince.datasets.load_hearthstone_cards()
        active = self.dataset.copy()
        if self.sup_rows:
            active = active[:n_active_rows]
        if self.sup_cols:
            active = active.drop(columns=["type_or_school"])
        self.ca = prince.MCA(n_components=n_components, engine="scipy")
        self.ca.fit(active)

        # Fit FactoMineR
        R("library('FactoMineR')")
        with tempfile.NamedTemporaryFile() as fp:
            self.dataset.to_csv(fp)
            R(f"dataset <- read.csv('{fp.name}')[,-1]")

        args = f"dataset, ncp={n_components}, graph=F"
        if self.sup_cols:
            if self.sup_rows:
                R(
                    f"ca <- MCA({args}, quali.sup=c(4), ind.sup=c({n_active_rows + 1}:nrow(dataset)))"
                )
            else:
                R(f"ca <- MCA({args}, quali.sup=c(4))")
        else:
            if self.sup_rows:
                R(f"ca <- MCA({args}, ind.sup=c({n_active_rows + 1}:nrow(dataset)))")
            else:
                R(f"ca <- MCA({args})")

    @pytest.mark.parametrize("method_name", ("row_coordinates", "transform"))
    def test_row_coords(self, method_name):
        super().test_row_coords(method_name=method_name)

    def test_col_coords(self):
        if self.sup_cols:
            F = load_df_from_R("ca$var$coord")
            if self.sup_cols:
                F = pd.concat((F, load_df_from_R("ca$quali.sup$coord")))
            P = self.ca.column_coordinates(self.dataset)
            np.testing.assert_allclose(F.abs(), P.abs())
        else:
            super().test_col_coords()

    def test_col_cos2(self):
        if self.sup_cols:
            F = load_df_from_R("ca$var$cos2")
            if self.sup_cols:
                F = pd.concat((F, load_df_from_R("ca$quali.sup$cos2")))
            P = self.ca.column_cosine_similarities(self.dataset)
            np.testing.assert_allclose(F, P)
        else:
            super().test_col_cos2()
